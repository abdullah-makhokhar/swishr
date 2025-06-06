# swishr - AI-Powered Basketball Shooting Analysis

## Project Overview

swishr is an innovative mobile application that transforms any smartphone into a professional basketball shooting coach. Using advanced computer vision technology (YOLO V8, MediaPipe) and machine learning, it provides real-time analysis of shooting form, trajectory, and performance metrics.

### Key Features

- **Real-time Shot Analysis**: Instant feedback on shooting form and trajectory
- **Professional-Grade Metrics**: Arc angle, release point consistency, follow-through analysis
- **Progress Tracking**: Detailed analytics and improvement trends over time
- **Personalized Coaching**: AI-generated recommendations based on individual weaknesses
- **Multi-Platform Support**: iOS and Android compatibility

## Demo

![Basketball Shot Analysis Demo](images/shot_detection_demo.png)

## Technical Architecture

### Core Technologies
- **Computer Vision**: YOLO V8, OpenCV, MediaPipe
- **Backend**: Python 3.9+, FastAPI, PostgreSQL, Redis
- **Machine Learning**: TensorFlow/PyTorch, Scikit-learn
- **Mobile**: React Native (cross-platform)
- **Cloud**: AWS/GCP with Docker containerization


## Project Setup

### Prerequisites
- Python 3.9 or higher
- Node.js 16+ (for mobile development)
- Git
- Docker (optional, for containerization)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdullah-makhokhar/swishr.git
   cd swishr
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   alembic upgrade head
   ```

### Development Setup

1. **Start the backend server**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Run tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

3. **Code formatting**
   ```bash
   black src/
   flake8 src/
   ```

## Project Structure

```
swishr/
├── src/
│   ├── computer_vision/      # Core CV algorithms
│   ├── analytics/           # Performance metrics and analysis
│   ├── models/             # Database models
│   ├── api/                # FastAPI endpoints
│   ├── services/           # Business logic
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── mobile/                 # React Native mobile app
├── docs/                   # Documentation
├── scripts/                # Deployment and utility scripts
└── data/                   # Training data and models
```

## Development Phases

### Phase 1: Foundation ✅
- [x] Project setup and environment configuration
- [x] Basic computer vision pipeline (YOLO V8 + MediaPipe)
- [x] Ball tracking and trajectory analysis
- [x] Shot detection algorithms
- [x] Database schema and data pipeline

### Phase 2: Advanced Analytics 
- [x] Biomechanical analysis engine
- [x] Performance metrics calculation
- [ ] AI feedback system
- [ ] Mobile optimization

### Phase 3: Mobile Development 
- [ ] React Native application
- [ ] Real-time camera integration
- [ ] User interface and experience
- [ ] Social features and progress tracking

### Phase 4: Testing & Deployment 
- [ ] Comprehensive testing suite
- [ ] Beta testing with users
- [ ] Production deployment
- [ ] App store publication

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:
- Email: abdullah.vocab@gmail.com
- GitHub Issues: [Create an issue](https://github.com/abdullah-makhokhar/swishr/issues)

---

**Built with ❤️ for the basketball community** 