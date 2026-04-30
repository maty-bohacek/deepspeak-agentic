#!/usr/bin/env python3
"""
Flask web server for on-demand Tavus conversation creation.
Creates conversation IDs only when users click "I'm Ready".
Logs all conversation creations to timestamped files.
"""
import os
import sys
import json
import yaml
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, request

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
PUBLIC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(PUBLIC_CONFIG_PATH, "r") as _f:
    _public_cfg = yaml.safe_load(_f) or {}
PUBLIC_DOMAIN = (_public_cfg.get("PUBLIC_DOMAIN", "INSERT_YOUR_PUBLIC_DOMAIN_HERE") or "").strip() or "INSERT_YOUR_PUBLIC_DOMAIN_HERE"

from tavus.tavus import TavusWrapper
from pipeline_paths import OUTPUT_DIR as WORK_ROOT, ensure_output_dirs

app = Flask(__name__, template_folder='templates', static_folder='static')

ensure_output_dirs()

# Configuration (all under data_collection/output/)
SCENARIOS_PATH = os.path.join(WORK_ROOT, "0_scenarios/scenarios.json")
COMBINATIONS_PATH = os.path.join(WORK_ROOT, "1_combinations/selected_combinations.txt")
CONVO_LOGS_DIR = os.path.join(WORK_ROOT, "3_output_videos/combo_url_convo_links")
HTML_CONVERSATIONS_DIR = os.path.join(WORK_ROOT, "2_html_conversations")

# Ensure critical directories exist (must happen at module load for gunicorn)
os.makedirs(HTML_CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(CONVO_LOGS_DIR, exist_ok=True)

# S3 Configuration for recording
S3_BUCKET_NAME = _public_cfg.get("TAVUS_S3_BUCKET", "")
S3_BUCKET_REGION = _public_cfg.get("TAVUS_S3_REGION", "us-east-1")
AWS_ASSUME_ROLE_ARN = _public_cfg.get("TAVUS_AWS_ASSUME_ROLE_ARN", "")

# Cache for scenarios and combinations
_scenarios_cache = None
_combos_cache = None

# Session tracking for logging
_current_session_dir = None


def get_session_log_dir():
    """Get or create the session log directory (date/timestamp subfolder)"""
    global _current_session_dir
    if _current_session_dir is None:
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        _current_session_dir = os.path.join(CONVO_LOGS_DIR, date_folder, timestamp)
        os.makedirs(_current_session_dir, exist_ok=True)
        print(f"📁 Created session log directory: {_current_session_dir}")
    return _current_session_dir


def log_conversation_creation(combo_id: str, start_url: str, conversation_id: str, 
                               conversation_url: str, persona_id: str, params: dict):
    """Log conversation creation to a text file"""
    log_dir = get_session_log_dir()
    
    # Find next available suffix for this combo_id
    base_filename = f"combo_{combo_id}"
    suffix = 1
    log_file = os.path.join(log_dir, f"{base_filename}_{suffix}.txt")
    
    # Increment suffix until we find an unused filename
    while os.path.exists(log_file):
        suffix += 1
        log_file = os.path.join(log_dir, f"{base_filename}_{suffix}.txt")
    
    timestamp = datetime.now().isoformat()
    
    log_content = f"""Conversation Creation Log
========================
Timestamp: {timestamp}
Combo ID: {combo_id}
Start URL: {start_url}
Conversation ID: {conversation_id}
Conversation URL (Daily.co): {conversation_url}
Persona ID: {persona_id}

Combo Parameters:
{json.dumps(params, indent=2)}
"""
    
    with open(log_file, 'w') as f:
        f.write(log_content)
    
    # Also append to a master log file for easy CSV conversion later
    master_log = os.path.join(log_dir, "_master_log.txt")
    with open(master_log, 'a') as f:
        f.write(f"{timestamp}\t{combo_id}\t{start_url}\t{conversation_id}\t{conversation_url}\t{persona_id}\n")
    
    print(f"📝 Logged conversation: combo={combo_id} (suffix {suffix}), convo_id={conversation_id}, file={os.path.basename(log_file)}")
    return log_file


def load_scenarios():
    """Load scenarios from JSON file"""
    global _scenarios_cache
    if _scenarios_cache is None:
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            _scenarios_cache = json.load(f)
    return _scenarios_cache


def load_combinations():
    """Load combinations from file"""
    global _combos_cache
    if _combos_cache is None:
        _combos_cache = {}
        with open(COMBINATIONS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                combo_id, rest = line.split(":", 1)
                combo_id = combo_id.strip()
                params = {}
                for part in rest.split(","):
                    if "=" in part:
                        key, value = part.strip().split("=", 1)
                        params[key.strip()] = value.strip()
                _combos_cache[combo_id] = params
    return _combos_cache


def create_persona_for_combo(tavus, combo_id, params, scenarios):
    """Create a persona for the given combo"""
    replica_id = params.get("default-replica-id") or params.get("replica_id")
    scenario_id = params.get("scenario", "")
    
    system_prompt = scenarios.get(scenario_id, {}).get("llm", "")
    if not system_prompt:
        raise ValueError(f"No LLM prompt found for scenario {scenario_id}")
    
    result = tavus.create_persona(
        persona_name=f"ondemand_scenario_{scenario_id}_combo_{combo_id}",
        system_prompt=system_prompt,
        context=system_prompt,
        default_replica_id=replica_id,
        llm_model="tavus-gpt-4o-mini"
    )
    
    if not result.get("success"):
        raise Exception(f"Failed to create persona: {result.get('error')}")
    
    return result["data"]["persona_id"]


@app.route('/')
def index():
    """Home page"""
    return "Tavus Conversation Server is running!"


@app.route('/start/<combo_id>')
def landing_page(combo_id):
    """
    Landing page where user clicks "I'm Ready" to start conversation.
    This page does NOT create the conversation yet.
    """
    combos = load_combinations()
    scenarios = load_scenarios()
    
    if combo_id not in combos:
        return f"Error: Combo {combo_id} not found", 404
    
    params = combos[combo_id]
    scenario_id = params.get("scenario", "")
    human_instructions = scenarios.get(scenario_id, {}).get("human", "")
    
    return render_template('landing.html', 
                         combo_id=combo_id,
                         human_instructions=human_instructions)


@app.route('/api/create-conversation/<combo_id>', methods=['POST'])
def create_conversation_api(combo_id):
    """
    API endpoint to create conversation ON DEMAND.
    Called when user clicks "I'm Ready" button.
    Creates the conversation and writes the HTML file, then returns redirect URL.
    """
    try:
        combos = load_combinations()
        scenarios = load_scenarios()
        
        if combo_id not in combos:
            return jsonify({"error": f"Combo {combo_id} not found"}), 404
        
        params = combos[combo_id]
        tavus = TavusWrapper()
        
        # Create persona
        persona_id = create_persona_for_combo(tavus, combo_id, params, scenarios)
        
        # S3 config for recording
        s3_config = {
            "aws_assume_role_arn": AWS_ASSUME_ROLE_ARN,
            "recording_s3_bucket_name": S3_BUCKET_NAME,
            "recording_s3_bucket_region": S3_BUCKET_REGION,
        }
        
        # Create conversation with timeouts starting NOW
        result = tavus.start_conversation(
            persona_id=persona_id,
            conversation_name=f"ondemand_combo_{combo_id}",
            max_call_duration=1200,  # 20 minutes from NOW
            participant_left_timeout=0,  # End immediately when user leaves
            participant_absent_timeout=600,  # 10 min grace period (user just clicked ready)
            enable_recording=True,
            s3_config=s3_config,
        )
        
        if not result.get("success"):
            return jsonify({"error": f"Failed to create conversation: {result.get('error')}"}), 500
        
        conversation_data = result["data"]
        conversation_url = conversation_data["conversation_url"]
        conversation_id = conversation_data["conversation_id"]
        
        # Generate and write the HTML file
        html_content = generate_conversation_html(conversation_url)
        html_filename = f"conversation_combo_{combo_id}.html"
        
        html_filepath = os.path.join(HTML_CONVERSATIONS_DIR, html_filename)
        os.makedirs(HTML_CONVERSATIONS_DIR, exist_ok=True)
        with open(html_filepath, 'w') as f:
            f.write(html_content)
        
        print(f"✅ HTML saved to: {html_filepath}")
        
        # Log the conversation creation
        start_url = f"https://{PUBLIC_DOMAIN}/start/{combo_id}"
        log_conversation_creation(
            combo_id=combo_id,
            start_url=start_url,
            conversation_id=conversation_id,
            conversation_url=conversation_url,
            persona_id=persona_id,
            params=params
        )
        
        # Return the static file URL (served by nginx)
        return jsonify({
            "success": True,
            "redirect_url": f"/conversation/{html_filename}",
            "conversation_url": conversation_url,
            "conversation_id": conversation_id,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/conversation/<filename>')
def serve_conversation_html(filename):
    """Serve conversation HTML files from data_collection/output/2_html_conversations."""
    import os.path
    from flask import send_file
    
    filepath = os.path.join(HTML_CONVERSATIONS_DIR, filename)
    
    if not os.path.exists(filepath):
        return f"Conversation file not found: {filename}", 404
    
    return send_file(filepath, mimetype='text/html')


def generate_conversation_html(conversation_url: str) -> str:
    """Generate the HTML content for the conversation page"""
    return f'''<!DOCTYPE html>
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
            background-color: #1a1a2e;
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
            color: white;
            font-size: 18px;
            text-align: center;
        }}
        .spinner {{
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid #2F80ED;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="conversation-container">
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Connecting to conversation...</div>
        </div>
    </div>
    
    <script>
        const call = window.Daily.createFrame({{
            showLeaveButton: true,
            lang: "en",
            showFullscreenButton: true,
            iframeStyle: {{
                position: 'fixed',
                top: '0',
                left: '0',
                width: '100%',
                height: '100%',
            }},
            theme: {{
                colors: {{
                    accent: "#2F80ED",
                    background: "#1a1a2e",
                    baseText: "#ffffff",
                }},
            }},
        }});
        
        call.on('joined-meeting', async () => {{
            document.getElementById('loading').style.display = 'none';
            try {{
                console.log('Starting automatic recording...');
                await call.startRecording();
                console.log('Recording started successfully');
            }} catch (error) {{
                console.error('Failed to start recording:', error);
            }}
        }});
        
        call.on('error', (error) => {{
            console.error('Call error:', error);
            document.getElementById('loading').innerHTML = 
                '<div style="color: #fc8181;">Error connecting. Please refresh the page.</div>';
        }});
        
        call.on('left-meeting', async () => {{
            try {{
                await call.stopRecording();
            }} catch (error) {{
                console.error('Failed to stop recording:', error);
            }}
        }});
        
        call.join({{ url: '{conversation_url}', maxParticipants: 2 }});
        
        window.addEventListener('beforeunload', async () => {{
            try {{
                await call.stopRecording();
            }} catch (error) {{}}
            call.leave();
        }});
    </script>
</body>
</html>'''


if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    # Ensure output directories exist
    os.makedirs(CONVO_LOGS_DIR, exist_ok=True)
    os.makedirs(HTML_CONVERSATIONS_DIR, exist_ok=True)
    
    print("="*80)
    print("Starting Tavus Conversation Server...")
    print("="*80)
    print(f"Output root: {WORK_ROOT}")
    print(f"Scenarios:   {SCENARIOS_PATH}")
    print(f"Combos:      {COMBINATIONS_PATH}")
    print(f"Logs:        {CONVO_LOGS_DIR}")
    print(f"HTML:        {HTML_CONVERSATIONS_DIR}")
    print()
    print(f"Landing pages:    http://localhost:5000/start/<combo_id>")
    print(f"Conversation API: POST http://localhost:5000/api/create-conversation/<combo_id>")
    print(f"HTML files:       http://localhost:5000/conversation/<filename>")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

