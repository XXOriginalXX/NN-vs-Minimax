# Chess AI: Neural Network vs Minimax

### Overview
An interactive chess simulation pitting two AI approaches against each other: classical minimax algorithm versus neural network-based decision making. Watch as these two different AI strategies compete in real-time chess matches.

### Live Demo
Experience the battle of AI strategies in real-time:
- **Web Version**: [https://nn-vs-minimax.onrender.com/](https://nn-vs-minimax.onrender.com/)
- **Terminal Version**: Available in this repository (see Installation instructions)

### Features
- **Two AI Implementations**:
  - Classical Minimax with Alpha-Beta Pruning
  - Neural Network-based evaluation and decision making
- **Interactive Visualization**:
  - Web interface with animated chess board
  - Terminal-based visualization option for lighter resource usage
- **Configurable Parameters**:
  - Adjustable search depth for Minimax
  - Customizable neural network architecture

<iframe width="560" height="315" src="https://youtu.be/PbY9SGMOcZU" frameborder="0" allowfullscreen></iframe>


### Technologies Used
- Python
- Chess library
- PyTorch/TensorFlow for neural network implementation
- Flask/FastAPI for web version

### Installation

#### Prerequisites
```bash
# Clone the repository
git clone https://github.com/yourusername/chess-ai-comparison.git
cd chess-ai-comparison

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Terminal Version
```bash
python chess_simulation.py
```

#### Web Version (Local)
```bash
python app.py
```
Then navigate to `http://localhost:5000` in your browser.

### How It Works

#### Minimax Agent
The minimax algorithm searches through the game tree to a specified depth, evaluating positions using a traditional chess evaluation function considering:
- Material balance
- Piece positioning
- King safety
- Pawn structure
- Control of center

The implementation includes alpha-beta pruning to optimize search efficiency.

#### Neural Network Agent
The neural network model was trained on thousands of high-level chess games and evaluates board positions based on:
- Board representation as input features
- Multiple convolutional/dense layers to process spatial relationships
- Position evaluation output that guides move selection

### Project Structure
```
├── app.py                  # Web application entry point
├── chess_simulation.py     # Terminal-based simulation entry 
├── static/                 # Web assets
├── templates/              # HTML templates
```

### Performance Analysis
Initial testing shows interesting trade-offs between the two approaches:
- **Minimax**: Excels in tactical positions with clear winning sequences
- **Neural Network**: Better at strategic, long-term positional play

Detailed benchmarking is available in the `analysis` directory.

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Acknowledgments
- Chess programming community for evaluation function insights


---

**Note**: This project is for educational purposes to compare different AI approaches to chess.
