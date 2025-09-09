# 🏥 Retinal Scan Images Dataset

## Download Required!

The retinal images dataset (3,222 images, ~1.5GB) is distributed separately due to GitHub size limits.

### Download from Slack:
1. Go to **#capstone-data** channel
2. Download **`retinal_images.zip`**
3. Extract all files to THIS directory
4. Delete the zip file after extraction

### Verify Success:
After extraction, this directory should contain:
- 3,222 .png files
- Files like: `000c1434d8d7.png`, `001639a390f0.png`, etc.

### Check with Python:
```python
import os
images = [f for f in os.listdir('.') if f.endswith('.png')]
print(f"Found {len(images)} retinal images")
# Should print: Found 3222 retinal images
```

⚠️ **The Diabetic Retinopathy CNN model requires these images!**