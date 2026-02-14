# ai-trainer (Windows only)

---

The pip install command for all python files:

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

`pip install av numpy pandas tqdm Pillow opencv-python pyarrow`

---

How to run:

**Recording:** run dataset_recorder.py, specify your camera and play from the raw display of the nintendo switch (not the feedback/reply)

**Training:** run trainer.py, specify the number of epochs and all.

*Warning:* at this step (after training is finished), you should have a /checkpoints folder, if it doesn't contain a "full_inference_model.pt" then you need to generate it manually... If so, please contact the creator of the repo.*

**Inference:** Once there's a finished model, run and play with it. This script let's you play alongside the ai. The AI takes over the gameplay if user is inactive, and stop when the user press any inputs to play again. Another script with inputs display should be uploaded.
