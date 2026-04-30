"""heygen liveavatar api wrapper"""

import requests
import json
import os
import sys
import argparse
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlencode

API_KEY = os.environ.get("HEYGEN_API_KEY", "").strip()
DEFAULT_AVATAR_ID = "9a4f4b1f-86f9-4acf-9a37-b81c21ae95e4"
DEFAULT_VOICE_ID = "254ffe1e-c89f-430f-8c36-9e7611d310c0"
DEFAULT_CONTEXT_ID = "c401b5b4-14df-4e57-b040-30ef43af24b8"
SERVER_BASE_URL = os.environ.get("HEYGEN_SERVER_URL", "http://localhost:3000")


class HeyGenLiveAvatarWrapper:
    """wrapper for heygen liveavatar api"""
    
    def __init__(self, api_key: str = None, server_base_url: str = None, debug: bool = False):
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.liveavatar.com/v1"
        self.server_base_url = server_base_url or SERVER_BASE_URL
        self.debug = debug
        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "X-API-KEY": self.api_key
        }
    
    def _debug_log(self, message: str):
        if self.debug:
            print(f"[debug] {message}")
    
    def create_session_token(
        self,
        mode: str = "FULL",
        avatar_id: str = None,
        voice_id: str = None,
        context_id: str = None,
        language: str = "en",
        first_sentence: str = None,
        session_duration_limit: int = None
    ) -> Dict[str, Any]:
        """create a session token"""
        url = f"{self.base_url}/sessions/token"
        
        data = {
            "mode": mode,
            "avatar_id": avatar_id or DEFAULT_AVATAR_ID
        }
        
        if mode == "FULL":
            data["avatar_persona"] = {
                "voice_id": voice_id or DEFAULT_VOICE_ID,
                "language": language
            }
            if context_id:
                data["avatar_persona"]["context_id"] = context_id
            if first_sentence:
                data["avatar_persona"]["first_sentence"] = first_sentence
        elif mode == "CUSTOM":
            data["voice_id"] = voice_id or DEFAULT_VOICE_ID
            data["language"] = language
        
        if session_duration_limit:
            data["session_duration_limit"] = session_duration_limit
        
        self._debug_log(f"creating session token: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, headers=self.headers, json=data)
        return self._handle_response(response)
    
    def start_session(self, session_token: str) -> Dict[str, Any]:
        """start a liveavatar session"""
        url = f"{self.base_url}/sessions/start"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {session_token}"
        }
        self._debug_log("starting session...")
        response = requests.post(url, headers=headers)
        return self._handle_response(response)
    
    def stop_session(self, session_token: str) -> Dict[str, Any]:
        """stop a liveavatar session"""
        url = f"{self.base_url}/sessions/stop"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {session_token}"
        }
        self._debug_log("stopping session...")
        response = requests.post(url, headers=headers)
        return self._handle_response(response)
    
    def create_conversation(
        self,
        mode: str = "FULL",
        avatar_id: str = None,
        voice_id: str = None,
        context_id: str = None,
        language: str = "en",
        first_sentence: str = None,
        session_duration_limit: int = 1800
    ) -> Dict[str, Any]:
        """create a complete conversation session"""
        token_result = self.create_session_token(
            mode=mode,
            avatar_id=avatar_id,
            voice_id=voice_id,
            context_id=context_id,
            language=language,
            first_sentence=first_sentence,
            session_duration_limit=session_duration_limit
        )
        
        if not token_result.get("success"):
            return token_result
        
        api_data = token_result["data"].get("data", token_result["data"])
        session_id = api_data.get("session_id")
        session_token = api_data.get("session_token")
        
        self._debug_log(f"session token created: {session_id}")
        
        start_result = self.start_session(session_token)
        
        if not start_result.get("success"):
            return start_result
        
        start_data = start_result["data"].get("data", start_result["data"])
        livekit_url = start_data.get("livekit_url") or start_data.get("url")
        livekit_token = start_data.get("livekit_client_token") or start_data.get("access_token")
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "session_token": session_token,
                "livekit_url": livekit_url,
                "livekit_token": livekit_token,
                "created_at": datetime.now().isoformat()
            }
        }
    
    def get_conversation_url(
        self,
        mode: str = "FULL",
        avatar_id: str = None,
        voice_id: str = None,
        context_id: str = None,
        language: str = "en",
        first_sentence: str = None,
        session_duration_limit: int = 1800
    ) -> str:
        """create a conversation and return a user-facing url"""
        conversation = self.create_conversation(
            mode=mode,
            avatar_id=avatar_id,
            voice_id=voice_id,
            context_id=context_id,
            language=language,
            first_sentence=first_sentence,
            session_duration_limit=session_duration_limit
        )
        
        if not conversation.get("success"):
            raise Exception(f"failed to create conversation: {conversation.get('error')}")
        
        data = conversation["data"]
        params = {
            "session_id": data["session_id"],
            "livekit_url": data["livekit_url"],
            "livekit_token": data["livekit_token"]
        }
        
        return f"{self.server_base_url}/index.html?{urlencode(params)}"
    
    def get_conversation_details(
        self,
        mode: str = "FULL",
        avatar_id: str = None,
        voice_id: str = None,
        context_id: str = None,
        language: str = "en",
        first_sentence: str = None,
        session_duration_limit: int = 1800
    ) -> Dict[str, Any]:
        """create a conversation and return full details"""
        conversation = self.create_conversation(
            mode=mode,
            avatar_id=avatar_id,
            voice_id=voice_id,
            context_id=context_id,
            language=language,
            first_sentence=first_sentence,
            session_duration_limit=session_duration_limit
        )
        
        if not conversation.get("success"):
            raise Exception(f"failed to create conversation: {conversation.get('error')}")
        
        data = conversation["data"]
        params = {
            "session_id": data["session_id"],
            "livekit_url": data["livekit_url"],
            "livekit_token": data["livekit_token"]
        }
        
        return {
            "url": f"{self.server_base_url}/index.html?{urlencode(params)}",
            "session_id": data["session_id"],
            "session_token": data["session_token"],
            "livekit_url": data["livekit_url"],
            "livekit_token": data["livekit_token"],
            "created_at": data["created_at"]
        }
    
    def list_avatars(self) -> Dict[str, Any]:
        """list available avatars"""
        url = f"{self.base_url}/avatars"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def list_voices(self) -> Dict[str, Any]:
        """list available voices"""
        url = f"{self.base_url}/voices"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def list_contexts(self) -> Dict[str, Any]:
        """list available contexts"""
        url = f"{self.base_url}/contexts"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """handle api response"""
        try:
            data = response.json()
        except ValueError:
            data = {"error": "invalid json response", "raw": response.text}
        
        self._debug_log(f"response status: {response.status_code}")
        if self.debug:
            self._debug_log(f"response: {json.dumps(data, indent=2) if isinstance(data, dict) else data}")
        
        return {
            "success": response.status_code in [200, 201],
            "status_code": response.status_code,
            "data": data if response.status_code in [200, 201] else None,
            "error": data if response.status_code not in [200, 201] else None
        }


heygen = HeyGenLiveAvatarWrapper()


def main():
    """command-line interface"""
    parser = argparse.ArgumentParser(description="create a heygen liveavatar session")
    
    parser.add_argument("--avatar_id", type=str, help="avatar id")
    parser.add_argument("--voice_id", type=str, help="voice id")
    parser.add_argument("--context_id", type=str, help="context id")
    parser.add_argument("--language", type=str, default="en", help="language code")
    parser.add_argument("--first_sentence", type=str, help="greeting message")
    parser.add_argument("--duration", type=int, default=1800, help="max duration in seconds")
    parser.add_argument("--mode", type=str, default="FULL", choices=["FULL", "CUSTOM"])
    parser.add_argument("--json", action="store_true", help="output json")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--server_url", type=str, help="server url")
    
    args = parser.parse_args()
    
    wrapper = HeyGenLiveAvatarWrapper(
        debug=args.debug,
        server_base_url=args.server_url if args.server_url else None
    )
    
    try:
        if args.json:
            details = wrapper.get_conversation_details(
                mode=args.mode,
                avatar_id=args.avatar_id,
                voice_id=args.voice_id,
                context_id=args.context_id,
                language=args.language,
                first_sentence=args.first_sentence,
                session_duration_limit=args.duration
            )
            print(json.dumps(details, indent=2))
        else:
            url = wrapper.get_conversation_url(
                mode=args.mode,
                avatar_id=args.avatar_id,
                voice_id=args.voice_id,
                context_id=args.context_id,
                language=args.language,
                first_sentence=args.first_sentence,
                session_duration_limit=args.duration
            )
            print(url)
            
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
