"""
Complete Dataset Collection System for Group Emotion Recognition
Meets ALL professor requirements:
- 2,000+ group photographs  (we actually target 4,000+)
- Diverse contexts (weddings, funerals, celebrations, meetings, sports, etc.)
- Demographics diversity (age, ethnicity, gender)
- Varied conditions (lighting, poses, occlusion)
- 5,000+ faces with bounding boxes
- Auto-labeling with face detection

Improved version:
- Uses MTCNN for better face detection (groups, small faces, side faces)
- Filters faces by size and sharpness to get CLEAR expressions
- Uses emotion-focused queries to get diverse emotions (happy, sad, angry, fear, disgust, surprise, neutral, etc.)
- All image filenames include 'emotiontry'
"""

import requests
from pathlib import Path
import time
import json
import cv2
import numpy as np
from datetime import datetime
import hashlib
import math
from mtcnn import MTCNN  # pip install mtcnn


class CompleteDatasetCollector:
    """
    Professional dataset collection system for group emotion recognition.
    Supports multiple API keys and multiple runs without duplication.
    """
    
    def __init__(self, output_dir="group_people_dataset_collection"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Each context tries to cover different emotions, not only happiness
        self.contexts = {
            # JOY / HAPPY / MIXED
            'weddings': (
                'wedding ceremony couple family guests happy smiling laughing emotional '
                'crying group people celebration love'
            ),

            'celebrations': (
                'birthday party celebration festival graduation laughing cheering excited '
                'happy people group'
            ),

            'concerts': (
                'concert audience music festival fans cheering excited happy surprised '
                'emotional crowd'
            ),

            'family': (
                'family reunion gathering dinner living room happy smiling laughing upset '
                'crying angry children parents group'
            ),

            # SADNESS
            'funerals': (
                'funeral memorial mourning sad crying upset people emotional faces group '
                'sad crying black dresses grief'
            ),

            'elderly_groups': (
                'elderly people seniors grandparents gathering meeting sad emotional '
                'serious neutral group faces'
            ),

            'care_home': (
                'nursing home elderly seniors therapy activity sad happy serious emotional '
                'interaction group'
            ),

            # SERIOUS / NEUTRAL / ANGER
            'meetings': (
                'business meeting team discussion argument conflict serious neutral stressed '
                'angry people office group'
            ),

            'education': (
                'classroom students teacher school exam test serious neutral bored stressed '
                'confused group students'
            ),

            'protests': (
                'protest rally demonstration shouting angry people serious emotional crowd'
            ),

            # SURPRISE
            'surprise_party': (
                'surprise party shocked faces surprised reaction emotional group celebration'
            ),

            # FEAR / HORROR
            'cinema': (
                'people watching horror movie kids watching scary movie frightened shocked '
                'fear reaction cinema dark theater group'
            ),

            'haunted_house': (
                'haunted house fear scared terrified jump scare dark scary attraction group'
            ),

            # SAFE FEAR / EMERGENCY
            'earthquake_aftermath': (
                'earthquake survivors rescue group panic scared crying confused shock '
                'emergency workers rubble volunteers frightened families'
            ),

            'rescue_operations': (
                'rescue team emergency responders firefighters rescuing people fear panic '
                'evacuation scared worried crying emotional group'
            ),

            'building_evacuation': (
                'fire drill building evacuation alarm people rushing running panic scared '
                'smoke exit fear reactions group'
            ),

            'disaster_drill': (
                'emergency drill simulation practice rescue training crowd panic reaction '
                'frightened scared group stress fear'
            ),

            # SPORTS (ANGER + JOY + SADNESS MIX)
            'sports': (
                'sports fans cheering celebrating excited happy angry disappointed sad '
                'supporters crowd stadium group'
            ),

            # DISGUST (RICH VARIETY)
            'bad_smell': (
                'people smelling something bad disgusting reaction covering nose bad odor '
                'rotten food garbage reaction group disgusted'
            ),

            'dirty_environment': (
                'dirty toilet filthy environment garbage trash pollution group reaction '
                'disgust nasty smell'
            ),

            'food_disgust': (
                'people tasting bad food spoiled food rotten food disgusting taste reaction '
                'wrinkled nose disgusted faces group'
            ),

            'garbage_smell': (
                'people smelling garbage bad odor trash bins foul smell covering nose '
                'disgust reaction group'
            ),

            'gross_reaction': (
                'people reacting to something gross shocking nasty disgusting reaction group '
                'cringing disgusted faces'
            ),

            'dirty_kitchen': (
                'dirty kitchen messy cooking area mold spoiled food disgusting smell group '
                'reaction covering nose hygiene'
            ),

            # DIVERSITY (AGE GROUPS)
            'children_groups': (
                'kids group children friends playing running classroom birthday school '
                'laughing sad crying surprised scared joy fear group'
            ),

            'family_kids': (
                'family with kids children parents playing picnic living room happy sad '
                'angry surprised mixed emotions group'
            ),

            # LIGHTING & OCCLUSION VARIATION
            'low_light_scenes': (
                'group night dark low light shadows dim lighting party outdoor evening '
                'faces partially visible occlusion'
            ),

            'bright_light_scenes': (
                'group bright light sunny outdoor harsh light backlit high contrast '
                'shadowed faces occlusion'
            ),

            'crowded_scenes': (
                'crowd group busy street people walking occlusion overlapping faces '
                'side faces motion blur'
            ),

            'profile_groups': (
                'group side view profile view people looking away turning heads partial '
                'faces occluded hidden expressions'
            ),
        }
        
        # Create folders
        for context in self.contexts.keys():
            (self.output_dir / context).mkdir(exist_ok=True)
        
        # Face detector (MTCNN)
        self.face_detector = MTCNN()
        
        # Statistics (will be overwritten if metadata exists)
        self.stats = {
            'total_downloaded': 0,
            'total_faces': 0,
            'by_context': {},
            'skipped': 0,
            'duplicates': 0
        }
        
        # Track downloaded images (avoid duplicates across runs)
        self.downloaded_hashes = set()
        
        # Metadata storage
        self.metadata = []
        
        # Try to load previous metadata & hashes (so reruns don't duplicate)
        self._load_existing_state()
    
    # ---------------------------
    # STATE LOADING
    # ---------------------------
    def _load_existing_state(self):
        """
        If dataset_metadata.json already exists, load:
        - previous metadata
        - previous statistics
        - previous image_hashes (for dedup across runs)
        """
        metadata_file = self.output_dir / "dataset_metadata.json"
        if not metadata_file.exists():
            return
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', [])
            self.stats = data.get('statistics', self.stats)

            # ensure keys exist
            self.stats.setdefault('total_downloaded', 0)
            self.stats.setdefault('total_faces', 0)
            self.stats.setdefault('by_context', {})
            self.stats.setdefault('skipped', 0)
            self.stats.setdefault('duplicates', 0)

            # Restore image hashes from metadata if present
            for m in self.metadata:
                img_hash = m.get('image_hash')
                if img_hash:
                    self.downloaded_hashes.add(img_hash)

            print("ℹ️ Loaded existing metadata and hashes (continuing without duplicates).")
        except Exception as e:
            print(f"⚠️ Could not load existing metadata: {e}")
    
    # ---------------------------
    # UTILS
    # ---------------------------
    def get_image_hash(self, image_path: Path) -> str:
        """Calculate hash to detect duplicates"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def detect_faces(self, image_path, min_face_area_ratio=0.01, min_focus=80):
        """
        Detect faces and return:
          - number of CLEAR faces
          - bounding boxes of clear faces

        A face is considered CLEAR if:
          - It is not too small (area ratio >= min_face_area_ratio)
          - It is not too blurry (variance of Laplacian >= min_focus)
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0, []

            height, width = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            detections = self.face_detector.detect_faces(img_rgb)
            clear_face_boxes = []

            for det in detections:
                if "box" not in det:
                    continue

                x, y, w, h = det["box"]

                # Fix negative coordinates from MTCNN
                x = max(0, x)
                y = max(0, y)
                w = max(1, w)
                h = max(1, h)

                # Skip faces that go outside the image
                if x + w > width or y + h > height:
                    continue

                face_img = img[y:y+h, x:x+w]

                # 1) Check face size (relative to whole image)
                face_area = w * h
                img_area = width * height
                if face_area / img_area < min_face_area_ratio:
                    # Too small → expression probably not clear
                    continue

                # 2) Check sharpness / focus (blur detection)
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                focus_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                if focus_measure < min_focus:
                    # Too blurry
                    continue

                clear_face_boxes.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "focus": float(focus_measure),
                })

            return len(clear_face_boxes), clear_face_boxes

        except Exception as e:
            print(f"Error detecting faces: {e}")
            return 0, []
    
    def check_image_quality(self, image_path):
        """Check image quality (resolution, brightness, etc.)"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Cannot read image"
            
            height, width = img.shape[:2]
            
            # Check minimum resolution
            if width < 300 or height < 300:
                return False, f"Too small: {width}x{height}"
            
            # Check if image is too dark or too bright
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 20:
                return False, "Too dark"
            if mean_brightness > 235:
                return False, "Too bright"
            
            return True, "OK"
            
        except Exception as e:
            return False, str(e)
    
    # ---------------------------
    # PEXELS SCRAPER (ONE KEY)
    # ---------------------------
    def scrape_pexels(self, api_key, target_per_context=35, min_faces=3):
        """
        Scrape from Pexels with context-aware collection for ONE api_key.
        target_per_context is *additional* images per context (not global).
        """
        base_url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": api_key}
        
        print("\n" + "="*60)
        print(f"🔍 PEXELS DATA COLLECTION (key: ****{api_key[-4:]})")
        print("="*60)
        
        for context, keywords in self.contexts.items():
            print(f"\n📂 Context: {context.upper()}")
            print(f"   Keywords: {keywords}")
            
            collected = 0
            page = 1
            
            # Initialize context stats if first time
            if context not in self.stats['by_context']:
                self.stats['by_context'][context] = {
                    'images': 0,
                    'faces': 0,
                    'skipped': 0
                }
            
            while collected < target_per_context:
                try:
                    params = {
                        'query': keywords,
                        'per_page': 80,
                        'page': page,
                        'orientation': 'landscape'  # Better for groups
                    }
                    
                    response = requests.get(base_url, headers=headers, params=params, timeout=10)
                    
                    if response.status_code != 200:
                        print(f"   ❌ API Error: {response.status_code} → {response.text}")
                        break
                    
                    data = response.json()
                    photos = data.get('photos', [])
                    
                    if not photos:
                        print(f"   ⚠️  No more results for '{context}' with this key.")
                        break
                    
                    for photo in photos:
                        if collected >= target_per_context:
                            break
                        
                        try:
                            img_url = photo['src']['large2x']
                            img_id = photo['id']
                            photographer = photo.get('photographer', 'unknown')
                            
                            # Download image
                            img_response = requests.get(img_url, timeout=10)
                            if img_response.status_code != 200:
                                continue
                            
                            # Save temporarily
                            temp_path = self.output_dir / context / f"temp_pexels_{img_id}.jpg"
                            with open(temp_path, 'wb') as f:
                                f.write(img_response.content)
                            
                            # Check for duplicates
                            img_hash = self.get_image_hash(temp_path)
                            if img_hash in self.downloaded_hashes:
                                temp_path.unlink()
                                self.stats['duplicates'] += 1
                                continue
                            
                            # Check quality
                            is_good, reason = self.check_image_quality(temp_path)
                            if not is_good:
                                temp_path.unlink()
                                self.stats['skipped'] += 1
                                self.stats['by_context'][context]['skipped'] += 1
                                continue
                            
                            # Detect CLEAR faces (size + sharpness)
                            face_count, face_boxes = self.detect_faces(temp_path)
                            
                            if face_count >= min_faces:
                                # Generate final filename with 'emotiontry' tag
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                final_name = f"pexels_emotiontry_{context}_{img_id}_{face_count}faces_{timestamp}.jpg"
                                final_path = self.output_dir / context / final_name
                                temp_path.rename(final_path)
                                
                                # Store metadata
                                metadata_entry = {
                                    'filename': final_name,
                                    'context': context,
                                    'source': 'pexels',
                                    'source_id': img_id,
                                    'url': photo.get('url'),
                                    'photographer': photographer,
                                    'num_faces': face_count,
                                    'face_boxes': face_boxes,
                                    'timestamp': timestamp,
                                    'image_hash': img_hash,
                                    'keywords': keywords
                                }
                                self.metadata.append(metadata_entry)
                                
                                collected += 1
                                self.stats['total_downloaded'] += 1
                                self.stats['total_faces'] += face_count
                                self.stats['by_context'][context]['images'] += 1
                                self.stats['by_context'][context]['faces'] += face_count
                                self.downloaded_hashes.add(img_hash)
                                
                                print(f"   ✅ [{collected}/{target_per_context}] {final_name} ({face_count} clear faces)")
                            else:
                                temp_path.unlink()
                                self.stats['skipped'] += 1
                                self.stats['by_context'][context]['skipped'] += 1
                            
                            time.sleep(0.3)  # Rate limiting
                            
                        except Exception as e:
                            print(f"   ⚠️  Error per-photo: {e}")
                            continue
                    
                    page += 1
                    time.sleep(1)  # Rate limiting between pages
                    
                except Exception as e:
                    print(f"   ❌ Page error: {e}")
                    break
            
            print(f"   📊 Collected with this key: {collected} images, {self.stats['by_context'][context]['faces']} faces (total for this context)")
            
            # Save metadata after each context
            self.save_metadata()
    
    # ---------------------------
    # UNSPLASH SCRAPER
    # ---------------------------
    def scrape_unsplash(self, access_key, target_per_context=35, min_faces=3):
        """
        Scrape from Unsplash.
        target_per_context is *additional* images per context (not global).
        """
        base_url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {access_key}"}
        
        print("\n" + "="*60)
        print("🔍 UNSPLASH DATA COLLECTION")
        print("="*60)
        
        for context, keywords in self.contexts.items():
            print(f"\n📂 Context: {context.upper()}")
            
            collected = 0
            page = 1
            
            if context not in self.stats['by_context']:
                self.stats['by_context'][context] = {
                    'images': 0,
                    'faces': 0,
                    'skipped': 0
                }
            
            while collected < target_per_context:
                try:
                    params = {
                        'query': keywords,
                        'per_page': 30,
                        'page': page,
                        'orientation': 'landscape'
                    }
                    
                    response = requests.get(base_url, headers=headers, params=params, timeout=10)
                    
                    if response.status_code != 200:
                        print(f"   ❌ API Error: {response.status_code} → {response.text}")
                        break
                    
                    data = response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        print(f"   ⚠️  No more results for '{context}' on Unsplash.")
                        break
                    
                    for photo in results:
                        if collected >= target_per_context:
                            break
                        
                        try:
                            img_url = photo['urls']['regular']
                            img_id = photo['id']
                            photographer = photo.get('user', {}).get('name', 'unknown')
                            
                            img_response = requests.get(img_url, timeout=10)
                            if img_response.status_code != 200:
                                continue
                            
                            temp_path = self.output_dir / context / f"temp_unsplash_{img_id}.jpg"
                            with open(temp_path, 'wb') as f:
                                f.write(img_response.content)
                            
                            img_hash = self.get_image_hash(temp_path)
                            if img_hash in self.downloaded_hashes:
                                temp_path.unlink()
                                self.stats['duplicates'] += 1
                                continue
                            
                            is_good, reason = self.check_image_quality(temp_path)
                            if not is_good:
                                temp_path.unlink()
                                self.stats['skipped'] += 1
                                self.stats['by_context'][context]['skipped'] += 1
                                continue
                            
                            face_count, face_boxes = self.detect_faces(temp_path)
                            
                            if face_count >= min_faces:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                final_name = f"unsplash_emotiontry_{context}_{img_id}_{face_count}faces_{timestamp}.jpg"
                                final_path = self.output_dir / context / final_name
                                temp_path.rename(final_path)
                                
                                metadata_entry = {
                                    'filename': final_name,
                                    'context': context,
                                    'source': 'unsplash',
                                    'source_id': img_id,
                                    'photographer': photographer,
                                    'num_faces': face_count,
                                    'face_boxes': face_boxes,
                                    'timestamp': timestamp,
                                    'image_hash': img_hash,
                                    'keywords': keywords
                                }
                                self.metadata.append(metadata_entry)
                                
                                collected += 1
                                self.stats['total_downloaded'] += 1
                                self.stats['total_faces'] += face_count
                                self.stats['by_context'][context]['images'] += 1
                                self.stats['by_context'][context]['faces'] += face_count
                                self.downloaded_hashes.add(img_hash)
                                
                                print(f"   ✅ [{collected}/{target_per_context}] {final_name}")
                            else:
                                temp_path.unlink()
                                self.stats['skipped'] += 1
                                self.stats['by_context'][context]['skipped'] += 1
                            
                            time.sleep(0.5)
                            
                        except Exception as e:
                            print(f"   ⚠️  Error per-photo: {e}")
                            continue
                    
                    page += 1
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"   ❌ Page error: {e}")
                    break
            
            print(f"   📊 Collected from Unsplash: {collected} images for this context")
            self.save_metadata()
    
    # ---------------------------
    # METADATA & REPORT
    # ---------------------------
    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'statistics': self.stats,
                'timestamp': datetime.now().isoformat(),
                'total_images': len(self.metadata),
                'total_faces': sum(m['num_faces'] for m in self.metadata)
            }, f, indent=2)
    
    def print_final_stats(self):
        """Print comprehensive statistics"""
        print("\n" + "="*60)
        print("📊 FINAL DATASET STATISTICS")
        print("="*60)
        
        print(f"\n✅ Total Images: {self.stats['total_downloaded']}")
        print(f"✅ Total Faces: {self.stats['total_faces']}")
        print(f"✅ Average Faces/Image: {self.stats['total_faces']/max(self.stats['total_downloaded'], 1):.1f}")
        print(f"⚠️  Skipped (quality/faces): {self.stats['skipped']}")
        print(f"⚠️  Duplicates removed: {self.stats['duplicates']}")
        
        print("\n📂 By Context:")
        for context, data in sorted(self.stats['by_context'].items()):
            print(f"   {context:20} → {data['images']:4} images, {data['faces']:5} faces")
        
        print(f"\n💾 Metadata saved: {self.output_dir / 'dataset_metadata.json'}")
        print(f"📁 Images saved under: {self.output_dir}/")
        
        # Check if requirements are met
        print("\n✅ REQUIREMENTS CHECK:")
        if self.stats['total_downloaded'] >= 2000:
            print(f"   ✅ Images: {self.stats['total_downloaded']} ≥ 2,000 required")
        else:
            print(f"   ⚠️  Images: {self.stats['total_downloaded']} < 2,000 required (need {2000 - self.stats['total_downloaded']} more)")
        
        if self.stats['total_faces'] >= 5000:
            print(f"   ✅ Faces: {self.stats['total_faces']} ≥ 5,000 required")
        else:
            print(f"   ⚠️  Faces: {self.stats['total_faces']} < 5,000 required (need {5000 - self.stats['total_faces']} more)")
        
        print("="*60)

    def save_collection_report(self):
        """Save human-readable collection report"""
        report_path = self.output_dir / "collection_report.txt"
        with open(report_path, 'w') as f:
            # Header
            f.write("="*60 + "\n")
            f.write("DATASET COLLECTION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Timestamp
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary Statistics
            f.write("SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Images: {self.stats['total_downloaded']}\n")
            f.write(f"Total Clear Faces: {self.stats['total_faces']}\n")
            avg_faces = self.stats['total_faces']/max(self.stats['total_downloaded'], 1)
            f.write(f"Average Faces/Image: {avg_faces:.2f}\n")
            f.write(f"Skipped (quality): {self.stats['skipped']}\n")
            f.write(f"Duplicates removed: {self.stats['duplicates']}\n\n")
            
            # Requirements Check
            f.write("REQUIREMENTS CHECK\n")
            f.write("-" * 60 + "\n")
            f.write(f"Images requirement (2,000): {'PASS' if self.stats['total_downloaded'] >= 2000 else 'FAIL'}\n")
            f.write(f"Faces requirement (5,000): {'PASS' if self.stats['total_faces'] >= 5000 else 'FAIL'}\n\n")
            
            # Context Breakdown
            f.write("BREAKDOWN BY CONTEXT\n")
            f.write("-" * 60 + "\n")
            for context, data in sorted(self.stats['by_context'].items()):
                f.write(f"  {context:20} {data['images']:4} images, {data['faces']:5} faces\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"📄 Report saved: {report_path}")


# ============================================
# MAIN EXECUTION (ALL KEYS, TARGET 4000+)
# ============================================

if __name__ == "__main__":
    
    print("="*60)
    print("GROUP EMOTION RECOGNITION - DATASET COLLECTOR")
    print("Target: 4,000+ images with 3+ CLEAR faces each context-aware")
    print("="*60)
    
    # Initialize collector (will load previous metadata if exists)
    collector = CompleteDatasetCollector(output_dir="group_people_dataset_collection")
    
    # ===============================
    # CONFIGURE YOUR API KEYS HERE
    # ===============================
    # 🔐 Put your real Pexels keys here:
    PEXELS_API_KEYS = [
        "YOUR_PEXELS_KEY_1",
        "YOUR_PEXELS_KEY_2",

    ]
    
    # 🔐 Unsplash key:
    UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_KEY"
    
    # ===============================
    # GLOBAL TARGET: 4,000+ IMAGES
    # ===============================
    TOTAL_TARGET_IMAGES = 4000
    
    # Count how many APIs are actually active
    active_apis = len([k for k in PEXELS_API_KEYS if k and not k.startswith("YOUR_PEXELS_KEY")])
    if UNSPLASH_ACCESS_KEY and not UNSPLASH_ACCESS_KEY.startswith("YOUR_UNSPLASH"):
        active_apis += 1
    
    if active_apis == 0:
        print("⚠️ No valid API keys configured. Please set Pexels/Unsplash keys.")
    else:
        total_contexts = len(collector.contexts)
        # images per API
        per_api_target = math.ceil(TOTAL_TARGET_IMAGES / active_apis)
        # images per context per API
        per_context_target = math.ceil(per_api_target / total_contexts)
        
        print(f"\n🎯 Total target images: {TOTAL_TARGET_IMAGES}")
        print(f"🔑 Active APIs: {active_apis}")
        print(f"📌 Per API target ≈ {per_api_target} images")
        print(f"📂 Per context per API ≈ {per_context_target} images\n")
        
        # -------------------------------------
        # Run Pexels for each valid key
        # -------------------------------------
        for key in PEXELS_API_KEYS:
            if not key or key.startswith("YOUR_PEXELS_KEY"):
                continue
            print(f"\n🚀 Starting Pexels collection with key ****{key[-4:]} ...")
            collector.scrape_pexels(
                api_key=key,
                target_per_context=per_context_target,
                min_faces=3
            )
        
        # -------------------------------------
        # Run Unsplash if key is set
        # -------------------------------------
        if UNSPLASH_ACCESS_KEY and not UNSPLASH_ACCESS_KEY.startswith("YOUR_UNSPLASH"):
            print("\n🚀 Starting Unsplash collection...")
            collector.scrape_unsplash(
                access_key=UNSPLASH_ACCESS_KEY,
                target_per_context=per_context_target,
                min_faces=3
            )
        else:
            print("ℹ️ Unsplash key not set or placeholder, skipping Unsplash.")
    
    # Print final statistics and save report
    collector.print_final_stats()
    collector.save_collection_report()
    
    print("\n✅ DATA COLLECTION COMPLETE!")
    print("\n📝 Next steps:")
    print("   1. Review dataset_metadata.json")
    print("   2. Optionally run MTCNN/RetinaFace again for tighter crops")
    print("   3. Apply your emotion recognition model for auto-labeling")
    print("   4. Start manual/assisted annotation if needed")
