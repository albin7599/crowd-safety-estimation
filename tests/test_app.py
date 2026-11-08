"""Tests for IADL-CDE Web Application: /health, endpoints, and synthetic video upload flow."""
import os
import io
import time
import tempfile
import unittest
import numpy as np
import cv2

from app import app
from jobs import get_job


class TestIADLCDE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        """Check /health returns device and model_loaded status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("device", data)
        self.assertTrue(data.get("model_loaded"))
        self.assertIn("active_model", data)

    def test_index_page(self):
        """Check GET / renders the upload page."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"IADL-CDE", response.data)
        self.assertIn(b"Calibration", response.data)

    def test_camera_page(self):
        """Check GET /camera renders adaptation page."""
        response = self.client.get("/camera")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Camera Adaptation", response.data)

    def test_fake_video_upload_and_analysis_flow(self):
        """Generate a short synthetic video, upload it, and verify processing."""
        # 1. Create synthetic video in memory/tempfile
        # 15 frames of 384x288 moving square
        temp_dir = tempfile.mkdtemp()
        fake_video_path = os.path.join(temp_dir, "test_clip.mp4")
        
        w, h, fps = 288, 192, 10.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(fake_video_path, fourcc, fps, (w, h))
        
        for i in range(15):
            frame = np.full((h, w, 3), 40, dtype=np.uint8)
            # Add a moving white box (synthetic motion)
            x = 20 + i * 10
            y = 50
            cv2.rectangle(frame, (x, y), (x + 30, y + 30), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        with open(fake_video_path, "rb") as f:
            video_bytes = f.read()

        # 2. Upload video with calib_seconds = 0 (using stored thresholds)
        data = {
            "video": (io.BytesIO(video_bytes), "test_clip.mp4"),
            "calib_seconds": "0",
            "calib_normal": "false"
        }
        
        upload_resp = self.client.post(
            "/upload",
            data=data,
            content_type="multipart/form-data",
            headers={"X-Requested-With": "XMLHttpRequest"}
        )
        self.assertEqual(upload_resp.status_code, 200)
        upload_data = upload_resp.get_json()
        self.assertIn("job_id", upload_data)
        job_id = upload_data["job_id"]

        # 3. Verify watch page is accessible
        watch_resp = self.client.get(f"/watch/{job_id}")
        self.assertEqual(watch_resp.status_code, 200)

        # 4. Wait for processing to complete or check stats
        max_wait = 15
        start = time.time()
        final_stats = None
        while time.time() - start < max_wait:
            stats_resp = self.client.get(f"/stats/{job_id}")
            self.assertEqual(stats_resp.status_code, 200)
            final_stats = stats_resp.get_json()
            if final_stats["status"] in ("done", "error"):
                break
            time.sleep(0.5)

        self.assertIsNotNone(final_stats)
        self.assertIn(final_stats["status"], ("done", "running", "queued"))

        # 5. Verify CSV endpoint
        csv_resp = self.client.get(f"/stats/{job_id}/csv")
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn(b"frame,time_sec,density", csv_resp.data)

        # Cleanup
        try:
            os.remove(fake_video_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
