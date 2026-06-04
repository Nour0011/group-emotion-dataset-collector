# Group Emotion Dataset Collector

An automated AI data pipeline for collecting and curating group emotion datasets. Built to feed a CNN and Vision Language Model (VLM) that analyzes facial expressions and scene context in group photographs.

Scrapes group photographs from Pexels and Unsplash, filters images by face count and quality using MTCNN face detection, and organizes them by emotion context with full deduplication across runs.

---

## Project Overview

This tool was built to solve a real data collection bottleneck in a capstone project on group emotion recognition. Manually collecting 2,000+ group photos with diverse emotions, lighting conditions, and demographics was not feasible, so this pipeline automates the entire process.

The collector targets 4,000+ images across 26 emotion-aware contexts such as weddings, funerals, protests, haunted houses, and disaster drills — covering all 7 basic emotions: happy, sad, angry, fear, disgust, surprise, and neutral.

---

## Features

- Multi-source scraping from Pexels and Unsplash
- Supports multiple API keys to maximize collection speed
- MTCNN face detection to ensure each image contains clear, visible faces
- Face quality filtering by size and sharpness (blur detection)
- Image quality checks (resolution, brightness)
- MD5 hash-based deduplication across multiple runs
- Persistent state — stop and resume without re-downloading
- Context-aware keyword queries for emotion diversity
- Auto-saves metadata and collection report after each context

---

## Emotion Contexts

| Category | Contexts |
|---|---|
| Happy / Joy | weddings, celebrations, concerts, family |
| Sadness | funerals, elderly_groups, care_home |
| Anger / Neutral | meetings, education, protests |
| Surprise | surprise_party |
| Fear | cinema, haunted_house, earthquake_aftermath, rescue_operations, building_evacuation, disaster_drill |
| Disgust | bad_smell, dirty_environment, food_disgust, garbage_smell, gross_reaction, dirty_kitchen |
| Diversity | children_groups, family_kids |
| Lighting / Occlusion | low_light_scenes, bright_light_scenes, crowded_scenes, profile_groups |

---

## Requirements

- Python 3.8+
- Pexels API key (free at https://www.pexels.com/api/)
- Unsplash API key (free at https://unsplash.com/developers)

Install dependencies:

```bash
pip install requests opencv-python numpy mtcnn
```

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/group-emotion-dataset-collector.git
cd group-emotion-dataset-collector
```

2. Open `collect_dataset.py` and add your API keys:

```python
PEXELS_API_KEYS = [
    "YOUR_PEXELS_KEY_1",
    "YOUR_PEXELS_KEY_2",
]

UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_KEY"
```

3. Run the collector:

```bash
python collect_dataset.py
```

---

## Output

```text
group_people_dataset_collection/
├── weddings/
├── funerals/
├── protests/
├── ... (one folder per context)
├── dataset_metadata.json
└── collection_report.txt
```

Each image filename includes the source, context, image ID, face count, and timestamp:

```text
pexels_emotiontry_weddings_12345_5faces_20240501_143022.jpg
```

The metadata JSON stores for each image:
- filename and context
- source and photographer
- number of clear faces
- face bounding boxes
- image hash for deduplication

---

## Dataset Targets

| Requirement | Target |
|---|---|
| Total images | 4,000+ |
| Total faces | 5,000+ |
| Min faces per image | 3 |
| Emotion contexts | 26 |

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core scripting |
| MTCNN | Face detection |
| OpenCV | Image quality checks, blur detection |
| Pexels API | Image source 1 |
| Unsplash API | Image source 2 |

---

## Use Case

Built as part of a capstone project on group emotion recognition using CNN and Vision Language Models (VLMs) to analyze facial expressions and scene context in group photographs.