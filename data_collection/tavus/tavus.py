import requests
import json
import os
import yaml
from typing import Optional, Dict, Any, List
from datetime import datetime

_config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if os.path.isfile(_config_path):
    with open(_config_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f) or {}
    API_KEY = (_cfg.get("TAVUS_API_KEY") or "").strip()
else:
    API_KEY = os.environ.get("TAVUS_API_KEY", "").strip()

DEFAULT_PERSONA_ID = os.environ.get("TAVUS_DEFAULT_PERSONA_ID", "").strip()


class TavusWrapper:
    """
    Comprehensive wrapper for Tavus Conversational Video Interface (CVI) API
    
    Features:
    - Create and manage personas
    - Start conversations with customizable settings
    - Generate embeddable HTML for conversations
    - Support for S3 recording
    - Webhook callbacks for transcripts
    - Call duration and timeout management
    """
    
    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        self.base_url = "https://tavusapi.com/v2"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
    
    def create_persona(
        self,
        persona_name: str,
        system_prompt: str,
        context: str = "",
        default_replica_id: str = "rfe12d8b9597",  # Nathan stock replica
        pipeline_mode: str = "full",
        perception_model: str = "raven-0",
        smart_turn_detection: bool = True,
        llm_model: str = "gpt-4o-mini",
        tts_voice: str = "79a125e8-cd45-4c13-8a67-188112f4dd22"
    ) -> Dict[str, Any]:
        """
        Create a new persona with full pipeline capabilities
        
        Args:
            persona_name: Name for the persona
            system_prompt: System prompt defining persona behavior
            context: Additional context for the persona
            default_replica_id: ID of the replica to use (default: Nathan)
            pipeline_mode: "full" or "echo" mode
            perception_model: Model for perception (default: raven-0)
            smart_turn_detection: Enable smart turn detection
            llm_model: LLM model to use
            tts_voice: TTS voice ID
        
        Returns:
            Dict containing persona creation response
        """
        url = f"{self.base_url}/personas"
        
        # Use the correct API format based on documentation
        data = {
            "pipeline_mode": pipeline_mode,
            "system_prompt": system_prompt
        }
        
        # Add optional fields if provided
        if persona_name:
            data["persona_name"] = persona_name
        if context:
            data["context"] = context
        if default_replica_id:
            data["default_replica_id"] = default_replica_id
        
        # Add layers configuration for LLM model and other settings
        layers = {}
        
        # LLM configuration
        if llm_model:
            layers["llm"] = {
                "model": llm_model
            }
        
        # STT configuration
        if smart_turn_detection is not None:
            layers["stt"] = {
                "smart_turn_detection": smart_turn_detection
            }
        
        # Perception configuration
        if perception_model and perception_model != "off":
            layers["perception"] = {
                "perception_model": perception_model
            }
        
        if layers:
            data["layers"] = layers
        
        response = requests.post(url, headers=self.headers, json=data)
        return self._handle_response(response)
    
    def start_conversation(
        self,
        persona_id: str = None,
        replica_id: str = None,
        conversation_name: str = None,
        conversational_context: str = "",
        callback_url: str = None,
        max_call_duration: int = 1200,  # 20 minutes
        participant_left_timeout: int = 0,
        participant_absent_timeout: int = 120, # FLAG 
        enable_recording: bool = False,
        s3_config: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation
        
        Args:
            persona_id: ID of persona to use (takes precedence over replica_id)
            replica_id: ID of replica to use if no persona_id provided
            conversation_name: Name for the conversation
            conversational_context: Context for the conversation
            callback_url: URL for webhook callbacks (for transcripts)
            max_call_duration: Maximum call duration in seconds
            participant_left_timeout: Timeout after participant leaves
            participant_absent_timeout: Timeout if no one joins
            enable_recording: Whether to enable recording
            s3_config: S3 configuration dict with keys: aws_assume_role_arn, 
                      recording_s3_bucket_region, recording_s3_bucket_name
        
        Returns:
            Dict containing conversation details including URL
        """
        url = f"{self.base_url}/conversations"
        
        # Optional default persona (set TAVUS_DEFAULT_PERSONA_ID or pass persona_id / replica_id)
        if not persona_id and not replica_id and DEFAULT_PERSONA_ID:
            persona_id = DEFAULT_PERSONA_ID
        
        data = {
            "conversation_name": conversation_name or f"Conversation {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "conversational_context": conversational_context,
            "properties": {
                "max_call_duration": max_call_duration,
                "participant_left_timeout": participant_left_timeout,
                "participant_absent_timeout": participant_absent_timeout
            }
        }
        
        # Add persona or replica ID
        if persona_id:
            data["persona_id"] = persona_id
        elif replica_id:
            data["replica_id"] = replica_id
        
        # Add callback URL if provided
        if callback_url:
            data["callback_url"] = callback_url
        
        # Add recording configuration if enabled
        if enable_recording and s3_config:
            data["properties"].update({
                "enable_recording": True,
                "aws_assume_role_arn": s3_config.get("aws_assume_role_arn"),
                "recording_s3_bucket_region": s3_config.get("recording_s3_bucket_region"),
                "recording_s3_bucket_name": s3_config.get("recording_s3_bucket_name")
            })
        elif enable_recording:
            data["properties"]["enable_recording"] = True
        
        response = requests.post(url, headers=self.headers, json=data)
        return self._handle_response(response)
    
    def get_conversation_link(
        self,
        persona_id: str = None,
        replica_id: str = None,
        conversation_name: str = None,
        conversational_context: str = "",
        callback_url: str = None,
        **kwargs
    ) -> str:
        """
        Start a conversation and return only the join link
        
        Returns:
            String URL for joining the conversation
        """
        result = self.start_conversation(
            persona_id=persona_id,
            replica_id=replica_id,
            conversation_name=conversation_name,
            conversational_context=conversational_context,
            callback_url=callback_url,
            **kwargs
        )
        
        if result.get("success") and result.get("data"):
            return result["data"].get("conversation_url")
        else:
            raise Exception(f"Failed to create conversation: {result}")
    
    def create_embedded_conversation(
        self,
        persona_id: str = None,
        replica_id: str = None,
        conversation_name: str = None,
        conversational_context: str = "",
        callback_url: str = None,
        ui_config: Dict[str, Any] = None,
        output_file: str = "conversation.html",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a conversation and generate an HTML file for embedding
        
        Args:
            ui_config: Dictionary with UI customization options:
                - show_leave_button: bool
                - show_fullscreen_button: bool
                - language: str (e.g., "jp", "en")
                - theme_colors: dict with accent, background, baseText
            output_file: Filename for the generated HTML file
            
        Returns:
            Dict with conversation details and HTML file path
        """
        # Create conversation
        conversation = self.start_conversation(
            persona_id=persona_id,
            replica_id=replica_id,
            conversation_name=conversation_name,
            conversational_context=conversational_context,
            callback_url=callback_url,
            **kwargs
        )
        
        if not conversation.get("success"):
            raise Exception(f"Failed to create conversation: {conversation}")
        
        conversation_url = conversation["data"]["conversation_url"]
        
        # Default UI configuration
        default_ui_config = {
            "show_leave_button": True,
            "show_fullscreen_button": True,
            "language": "en",
            "theme_colors": {
                "accent": "#2F80ED",
                "background": "#F8F9FA",
                "baseText": "#1A1A1A"
            }
        }
        
        if ui_config:
            default_ui_config.update(ui_config)
        
        # Generate HTML template
        html_content = self._generate_html_template(conversation_url, default_ui_config)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return {
            "success": True,
            "conversation": conversation["data"],
            "html_file": output_file,
            "html_content": html_content
        }
    
    def list_personas(self) -> Dict[str, Any]:
        """
        List all available personas
        
        Returns:
            Dict containing list of personas
        """
        url = f"{self.base_url}/personas"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def get_persona(self, persona_id: str) -> Dict[str, Any]:
        """
        Get details for a specific persona
        
        Args:
            persona_id: ID of the persona
            
        Returns:
            Dict containing persona details
        """
        url = f"{self.base_url}/personas/{persona_id}"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def delete_persona(self, persona_id: str) -> Dict[str, Any]:
        """
        Delete a persona
        
        Args:
            persona_id: ID of the persona to delete
            
        Returns:
            Dict containing deletion response
        """
        url = f"{self.base_url}/personas/{persona_id}"
        # Use only the API key header for DELETE requests as per documentation
        delete_headers = {"x-api-key": self.api_key}
        response = requests.delete(url, headers=delete_headers)
        return self._handle_response(response)
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get details for a specific conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dict containing conversation details
        """
        url = f"{self.base_url}/conversations/{conversation_id}"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def list_conversations(self) -> Dict[str, Any]:
        """
        List all conversations
        
        Returns:
            Dict containing list of conversations
        """
        url = f"{self.base_url}/conversations"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)
    
    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        End a conversation
        
        Args:
            conversation_id: ID of the conversation to end
            
        Returns:
            Dict containing response
        """
        url = f"{self.base_url}/conversations/{conversation_id}/end"
        response = requests.post(url, headers=self.headers)
        return self._handle_response(response)
    
    def _generate_html_template(self, conversation_url: str, ui_config: Dict[str, Any]) -> str:
        """
        Generate HTML template for embedding conversation
        
        Args:
            conversation_url: URL of the conversation
            ui_config: UI configuration options (includes enable_auto_recording)
            
        Returns:
            HTML content as string
        """
        theme_colors = ui_config.get("theme_colors", {})
        enable_auto_recording = ui_config.get("enable_auto_recording", False)
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tavus Conversation</title>
    <script crossorigin src="https://unpkg.com/@daily-co/daily-js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {theme_colors.get('background', '#F8F9FA')};
        }}
        .conversation-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: {theme_colors.get('baseText', '#1A1A1A')};
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="conversation-container">
        <div class="loading" id="loading">Loading conversation...</div>
    </div>
    
    <script>
        // Initialize Daily call frame
        const call = window.Daily.createFrame({{
            showLeaveButton: {str(ui_config.get('show_leave_button', True)).lower()},
            lang: "{ui_config.get('language', 'en')}",
            showFullscreenButton: {str(ui_config.get('show_fullscreen_button', True)).lower()},
            iframeStyle: {{
                position: 'fixed',
                top: '0',
                left: '0',
                width: '100%',
                height: '100%',
            }},
            theme: {{
                colors: {{
                    accent: "{theme_colors.get('accent', '#2F80ED')}",
                    background: "{theme_colors.get('background', '#F8F9FA')}",
                    baseText: "{theme_colors.get('baseText', '#1A1A1A')}",
                }},
            }},
        }});
        
        // Hide loading indicator once call starts
        call.on('joined-meeting', async () => {{
            document.getElementById('loading').style.display = 'none';
            
            {'// Automatically start recording' if enable_auto_recording else ''}
            {f'''try {{
                console.log('Starting automatic recording...');
                await call.startRecording();
                console.log('Recording started successfully');
            }} catch (error) {{
                console.error('Failed to start recording:', error);
            }}''' if enable_auto_recording else ''}
        }});
        
        // Handle errors
        call.on('error', (error) => {{
            console.error('Call error:', error);
            document.getElementById('loading').innerText = 'Error loading conversation. Please refresh the page.';
        }});
        
        {'// Stop recording when call ends' if enable_auto_recording else ''}
        {f'''call.on('left-meeting', async () => {{
            try {{
                console.log('Stopping recording...');
                await call.stopRecording();
                console.log('Recording stopped successfully');
            }} catch (error) {{
                console.error('Failed to stop recording:', error);
            }}
        }});''' if enable_auto_recording else ''}
        
        // Join the conversation
        call.join({{ url: '{conversation_url}' }});
        
        // Optional: Handle page unload
        window.addEventListener('beforeunload', async () => {{
            {'// Stop recording on page close' if enable_auto_recording else ''}
            {f'''try {{
                await call.stopRecording();
            }} catch (error) {{
                // Ignore errors on unload
            }}''' if enable_auto_recording else ''}
            call.leave();
        }});
    </script>
</body>
</html>"""
        
        return html_template
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and return standardized format
        
        Args:
            response: requests.Response object
            
        Returns:
            Standardized response dictionary
        """
        # Handle empty responses (common for DELETE operations)
        if not response.text.strip():
            return {
                "success": response.status_code in [200, 201, 204],  # 204 is No Content (successful DELETE)
                "status_code": response.status_code,
                "data": None if response.status_code in [200, 201, 204] else {"message": "Empty response"},
                "error": None if response.status_code in [200, 201, 204] else {"message": "Empty response"},
                "raw_response": response.text
            }
        
        try:
            data = response.json()
        except ValueError:
            data = {"error": "Invalid JSON response"}
        
        return {
            "success": response.status_code in [200, 201, 204],  # 204 is No Content (successful DELETE)
            "status_code": response.status_code,
            "data": data if response.status_code in [200, 201, 204] else None,
            "error": data if response.status_code not in [200, 201, 204] else None,
            "raw_response": response.text
        }


# Example usage and helper functions
def create_sales_coach_persona(tavus: TavusWrapper) -> Dict[str, Any]:
    """
    Example: Create a sales coach persona
    """
    return tavus.create_persona(
        persona_name="Sales Coach AI",
        system_prompt="You are an experienced sales coach who helps people improve their sales techniques. You provide constructive feedback, role-play scenarios, and share proven sales strategies.",
        context="You have 20+ years of experience in B2B sales and have trained thousands of sales professionals.",
        smart_turn_detection=True
    )

def create_interview_persona(tavus: TavusWrapper) -> Dict[str, Any]:
    """
    Example: Create an interviewer persona
    """
    return tavus.create_persona(
        persona_name="Technical Interviewer",
        system_prompt="You are a skilled technical interviewer who conducts thoughtful and structured interviews. Your aim is to ask insightful questions, listen carefully, and assess responses objectively.",
        context="You have experience interviewing candidates for software engineering, data science, and product management roles.",
        perception_model="raven-0",  # Enable screen sharing for technical interviews
        smart_turn_detection=True
    )

def quick_conversation(
    tavus: TavusWrapper,
    persona_name: str = "Quick Chat",
    context: str = "Have a casual conversation",
    callback_url: str = None
) -> str:
    """
    Quickly start a conversation with default settings
    
    Returns:
        Conversation URL
    """
    return tavus.get_conversation_link(
        conversation_name=persona_name,
        conversational_context=context,
        callback_url=callback_url
    )


# Initialize the wrapper
tavus = TavusWrapper()

# Stock personas and replicas available
STOCK_REPLICAS = {
    "nathan": "rfe12d8b9597",  # Default - Nathan
    "phoenix": "ra066ab28864"  # Phoenix-3
}

STOCK_PERSONAS = {
    "sales_coach": "pdced222244b"  # Default sales coach
}

