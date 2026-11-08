"""
Kaggle MCP Client for Automated Submissions, Notebook Execution, and Telemetry.

Integrates with the official Kaggle MCP server (https://www.kaggle.com/mcp)
using standard JSON-RPC protocol over HTTP.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

KAGGLE_MCP_URL = "https://www.kaggle.com/mcp"
DEFAULT_COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
DEFAULT_KERNEL_SLUG = "notebook0b6a5a21c6"
DEFAULT_USER = "heetp0"


class KaggleMCPClient:
    """Client for Kaggle Model Context Protocol (MCP) server."""

    def __init__(self, token: Optional[str] = None):
        if token is None:
            # 1. Try environment variable
            token = os.environ.get("KAGGLE_MCP_TOKEN")
        if token is None:
            # 2. Try ~/.gemini/settings.json
            settings_path = Path.home() / ".gemini" / "settings.json"
            if settings_path.exists():
                try:
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                        kaggle_srv = settings.get("mcpServers", {}).get("kaggle", {})
                        headers = kaggle_srv.get("headers", {})
                        auth = headers.get("Authorization", "")
                        if auth.startswith("Bearer "):
                            token = auth.split("Bearer ", 1)[1].strip()
                except Exception:
                    pass

        if not token:
            raise ValueError(
                "Kaggle MCP token not found. Provide via constructor, KAGGLE_MCP_TOKEN env var, "
                "or ~/.gemini/settings.json."
            )
        self.token = token
        self._request_id = 0

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on Kaggle MCP server via JSON-RPC."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            KAGGLE_MCP_URL,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "PhytoForge-KaggleMCP/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                # MCP SSE response format: 'event: message\ndata: {...}\n'
                for line in raw.splitlines():
                    if line.startswith("data:"):
                        line_data = line[len("data:") :].strip()
                        res_json = json.loads(line_data)
                        if "error" in res_json:
                            raise RuntimeError(f"MCP Error: {res_json['error']}")
                        result = res_json.get("result", {})
                        if result.get("isError"):
                            content_txt = result.get("content", [{}])[0].get("text", "")
                            raise RuntimeError(f"Tool execution failed: {content_txt}")
                        # If content is text JSON, parse it
                        content = result.get("content", [])
                        if content and content[0].get("type") == "text":
                            try:
                                return json.loads(content[0]["text"])
                            except json.JSONDecodeError:
                                return {"raw_text": content[0]["text"]}
                        return result
                raise RuntimeError(f"Unexpected response stream format: {raw[:200]}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} error from Kaggle MCP: {err_body}")

    def get_competition_status(self, competition_name: str = DEFAULT_COMPETITION) -> Dict[str, Any]:
        """Fetch metadata for competition."""
        return self.call_tool("get_competition", {"request": {"competitionName": competition_name}})

    def get_leaderboard(self, competition_name: str = DEFAULT_COMPETITION, page_size: int = 10) -> Dict[str, Any]:
        """Fetch top ranks from leaderboard."""
        return self.call_tool(
            "get_competition_leaderboard",
            {"request": {"competitionName": competition_name, "pageSize": page_size}},
        )

    def get_submissions(self, competition_name: str = DEFAULT_COMPETITION) -> List[Dict[str, Any]]:
        """Fetch historical submissions for the user."""
        res = self.call_tool("search_competition_submissions", {"request": {"competitionName": competition_name}})
        return res.get("submissions", [])

    def get_accelerator_quota(self) -> Dict[str, Any]:
        """Fetch remaining GPU/TPU hours."""
        return self.call_tool("get_accelerator_quota", {"request": {}})

    def get_notebook_info(self, kernel_slug: str = DEFAULT_KERNEL_SLUG, user_name: str = DEFAULT_USER) -> Dict[str, Any]:
        """Fetch notebook details and configuration."""
        return self.call_tool("get_notebook_info", {"request": {"kernelSlug": kernel_slug, "userName": user_name}})

    def push_kernel_code(
        self,
        script_code: str,
        kernel_slug: str = DEFAULT_KERNEL_SLUG,
        kernel_title: str = "PhytoForge CASMI 2026 Standalone",
        dataset_data_sources: Optional[List[str]] = None,
        competition_data_sources: Optional[List[str]] = None,
        run_after_save: bool = True,
    ) -> Dict[str, Any]:
        """
        Package Python script into Jupyter Notebook format and push to Kaggle via save_notebook.
        """
        if dataset_data_sources is None:
            dataset_data_sources = ["heetp0/phytoforge-assets"]
        if competition_data_sources is None:
            competition_data_sources = [DEFAULT_COMPETITION]

        # Construct single-cell notebook JSON
        notebook_dict = {
            "metadata": {
                "kernelspec": {
                    "language": "python",
                    "display_name": "Python 3",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.12",
                    "mimetype": "text/x-python",
                    "file_extension": ".py",
                },
            },
            "nbformat_minor": 4,
            "nbformat": 4,
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {"trusted": True},
                    "execution_count": None,
                    "outputs": [],
                    "source": script_code,
                }
            ],
        }
        notebook_text = json.dumps(notebook_dict)

        request_args: Dict[str, Any] = {
            "slug": kernel_slug,
            "newTitle": kernel_title,
            "text": notebook_text,
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": True,
            "enableGpu": False,
            "enableTpu": False,
            "enableInternet": False,
            "datasetDataSources": dataset_data_sources,
            "competitionDataSources": competition_data_sources,
            "kernelExecutionType": "SaveAndRunAll" if run_after_save else "QuickSave",
        }

        return self.call_tool("save_notebook", {"request": request_args})

    def get_notebook_session_status(
        self, kernel_slug: str = DEFAULT_KERNEL_SLUG, user_name: str = DEFAULT_USER
    ) -> Dict[str, Any]:
        """Fetch running status of the kernel session."""
        return self.call_tool(
            "get_notebook_session_status",
            {"request": {"kernelSlug": kernel_slug, "userName": user_name}},
        )

    def submit_code_version(
        self,
        kernel_slug: str = DEFAULT_KERNEL_SLUG,
        kernel_owner: str = DEFAULT_USER,
        file_name: str = "submission.csv",
        description: str = "PhytoForge v1.0.6 Submission",
        competition_name: str = DEFAULT_COMPETITION,
    ) -> Dict[str, Any]:
        """Submit the generated output of a kernel run directly to competition."""
        return self.call_tool(
            "create_code_competition_submission",
            {
                "request": {
                    "competitionName": competition_name,
                    "kernelOwner": kernel_owner,
                    "kernelSlug": kernel_slug,
                    "fileName": file_name,
                    "submissionDescription": description,
                }
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PhytoForge Kaggle MCP Controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status command
    subparsers.add_parser("status", help="Query competition status, leaderboard, and user quota")

    # push command
    push_parser = subparsers.add_parser("push", help="Push dist/kaggle_kernel.py to Kaggle and trigger SaveAndRunAll")
    push_parser.add_argument("--slug", type=str, default=DEFAULT_KERNEL_SLUG, help="Target kernel slug")
    push_parser.add_argument("--file", type=str, default="dist/kaggle_kernel.py", help="Kernel file to push")

    # poll command
    poll_parser = subparsers.add_parser("poll", help="Poll notebook execution status")
    poll_parser.add_argument("--slug", type=str, default=DEFAULT_KERNEL_SLUG, help="Target kernel slug")

    # submit command
    sub_parser = subparsers.add_parser("submit", help="Submit kernel output to competition")
    sub_parser.add_argument("--slug", type=str, default=DEFAULT_KERNEL_SLUG, help="Target kernel slug")
    sub_parser.add_argument("--desc", type=str, default="PhytoForge v1.0.6 Omega Submission", help="Submission description")

    args = parser.parse_args()
    client = KaggleMCPClient()

    if args.command == "status":
        print("=" * 60)
        print("  PhytoForge Kaggle MCP Controller: Live Status Query")
        print("=" * 60)

        # Quota
        quota = client.get_accelerator_quota()
        gpu = quota.get("gpu_quota", {})
        print(f"GPU Quota:  Allowed: {gpu.get('total_time_allowed')} | Used: {gpu.get('time_used')}")

        # Competition
        comp = client.get_competition_status()
        print(f"Competition: {comp.get('title')}")
        print(f"User Rank:   #{comp.get('user_rank')} / {comp.get('team_count')} teams")
        print(f"Max Submits: {comp.get('max_daily_submissions')} / day")

        # Submissions
        subs = client.get_submissions()
        print(f"\nRecent Submissions ({len(subs)} found):")
        for s in subs[:3]:
            print(f"  - [{s.get('date')}] Ref: {s.get('ref')} | Score: {s.get('public_score')} | Status: {s.get('status')}")

        # Top 5 Leaderboard
        lb = client.get_leaderboard(page_size=5)
        print("\nCurrent Leaderboard (Top 5):")
        for idx, entry in enumerate(lb.get("submissions", []), 1):
            print(f"  {idx}. {entry.get('team_name'):<25} | Score: {entry.get('score')} | Date: {entry.get('submission_date')}")

    elif args.command == "push":
        kernel_path = Path(args.file)
        if not kernel_path.exists():
            print(f"Error: file {kernel_path} does not exist. Run bundle builder first.")
            sys.exit(1)
        code = kernel_path.read_text(encoding="utf-8")
        print(f"Pushing {len(code):,} characters from {kernel_path} to kernel '{args.slug}'...")
        res = client.push_kernel_code(script_code=code, kernel_slug=args.slug)
        print("Kernel successfully uploaded and queued for execution!")
        print(json.dumps(res, indent=2))

    elif args.command == "poll":
        print(f"Polling session status for '{args.slug}'...")
        res = client.get_notebook_session_status(kernel_slug=args.slug)
        print(json.dumps(res, indent=2))

    elif args.command == "submit":
        print(f"Submitting kernel '{args.slug}' to competition...")
        res = client.submit_code_version(kernel_slug=args.slug, description=args.desc)
        print("Submission submitted successfully!")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
