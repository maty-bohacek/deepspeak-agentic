#!/bin/bash
# Run the Flask web server for on-demand Tavus conversation creation
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DC_ROOT"
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
cd "$SCRIPT_DIR"

echo "Starting Tavus On-Demand Conversation Server..."
echo ""
echo "URLs:"
echo "  Landing page: http://localhost:5000/start/<combo_id>"
echo "  API endpoint: POST http://localhost:5000/api/create-conversation/<combo_id>"
echo ""
echo "For production with nginx, configure proxy_pass to http://127.0.0.1:5000"
echo ""

python app.py

# For production, use gunicorn instead:
# gunicorn -w 4 -b 127.0.0.1:5000 app:app
