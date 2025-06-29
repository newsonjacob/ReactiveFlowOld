import cv2
import numpy as np

def draw_overlay(vis_img, frame_count, speed, state, sim_time,
                 smooth_L, smooth_C, smooth_R,
                 delta_L, delta_C, delta_R,
                 left_count, center_count, right_count,
                 good_old, flow_vectors,
                 in_grace=False):
    img = vis_img.copy()
    h, w = img.shape[:2]
    third = w // 3

    # Draw flow vectors
    for i, (p1, vec) in enumerate(zip(good_old, flow_vectors)):
        if i > 50:
            break
        x1, y1 = int(p1[0]), int(p1[1])
        vec = np.ravel(vec)  # flatten safely
        if vec.shape[0] >= 2:
            dx, dy = float(vec[0]), float(vec[1])
        else:
            dx, dy = 0.0, 0.0
        x2, y2 = int(x1 + dx), int(y1 + dy)
        cv2.arrowedLine(img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

    # Draw dividers
    cv2.line(img, (third, 0), (third, h), (255, 255, 255), 2)
    cv2.line(img, (2 * third, 0), (2 * third, h), (255, 255, 255), 2)

    # Zone labels
    label_y = h - 20
    cv2.putText(img, "Left", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Center", (third + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Right", (2 * third + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Magnitudes + features
    cv2.putText(img, f"L: {smooth_L:.1f} ({left_count})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"C: {smooth_C:.1f} ({center_count})", (third + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"R: {smooth_R:.1f} ({right_count})", (2 * third + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"ΔL: {delta_L:+.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"ΔC: {delta_C:+.2f}", (third + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"ΔR: {delta_R:+.2f}", (2 * third + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Status overlay
    cv2.putText(img, f"Frame: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Speed: {speed:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"State: {state}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Sim Time: {sim_time:.2f}s", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if in_grace:
        cv2.putText(img, "GRACE", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    return img
