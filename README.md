# Crowd Safety Management Web App

A real-time crowd monitoring application built with Streamlit, YOLO, and MediaPipe.

## Features

- Real-time crowd detection using AI models
- Automated email alerts when thresholds are exceeded
- User and admin authentication
- Backend API integration
- Webcam and video file support

## Quick Start

### Local Development (Full Features)

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Install full dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   - Local: http://localhost:8501
   - Network: http://your-ip:8501

### Streamlit Cloud Deployment (Minimal Features)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Connect your GitHub repository
   - **Important:** Use `requirements-cloud.txt` for dependencies
   - Deploy!

   **Note:** Cloud version will use dummy detection (shows 0 people) due to dependency limitations.

## Configuration

### Email Alerts (Admin Users)

Configure SMTP settings in the sidebar:
- **SMTP Host:** `smtp.gmail.com`
- **SMTP Port:** `587`
- **Username:** Your Gmail address
- **Password:** Gmail App Password (not regular password)
- **From Email:** Your Gmail address
- **Admin Email:** Where alerts should be sent

### Backend Integration

Enable backend connection in sidebar:
- **Backend URL:** `http://localhost:3000` (for local Node.js backend)
- **Enable Backend:** Check the checkbox

## Models

The app supports multiple detection models:
- **YOLO:** Primary person detection (auto-downloads weights)
- **MediaPipe:** Face detection fallback
- **Dummy:** Fallback when AI models unavailable (used in cloud deployment)

## Troubleshooting

### Webcam Issues
- Try different camera indices (0, 1, 2...)
- Check camera permissions
- Use V4L2 backend on Linux

### Email Issues
- Use Gmail App Passwords, not regular passwords
- Enable 2-factor authentication first
- Check SMTP settings

### Performance
- Adjust frame stride (process every Nth frame)
- Use lower resolution for better performance
- Close other applications

### Streamlit Cloud Issues
- Use `requirements-cloud.txt` for deployment
- App will work but show 0 people (dummy mode)
- All other features (UI, alerts, backend) work normally

## File Structure

```
CROWD_MANAGEMENT/
├── app.py                 # Main Streamlit application
├── model_loader.py        # AI model loading and inference
├── requirements.txt       # Full dependencies (local development)
├── requirements-cloud.txt # Minimal dependencies (cloud deployment)
├── packages.txt          # System dependencies (cloud)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Deployment Options

### Option 1: Local Development (Recommended)
- Use `requirements.txt` (full AI features)
- Real crowd detection with YOLO/MediaPipe
- All features work perfectly

### Option 2: Streamlit Cloud (Limited)
- Use `requirements-cloud.txt` (minimal dependencies)
- App works but shows 0 people (dummy detection)
- UI, alerts, and backend integration work
- Good for testing the interface

## License

MIT License - see LICENSE file for details.
