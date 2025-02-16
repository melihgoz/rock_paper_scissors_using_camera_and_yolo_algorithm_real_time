import cv2
import time
import numpy as np
from ultralytics import YOLO

def get_winner(gesture_p1, gesture_p2):
    if gesture_p1 == gesture_p2:
        return 'tie'
    if (gesture_p1 == 'rock' and gesture_p2 == 'scissors') or \
       (gesture_p1 == 'scissors' and gesture_p2 == 'paper') or \
       (gesture_p1 == 'paper' and gesture_p2 == 'rock'):
        return 'player1'
    else:
        return 'player2'

def main():
    model_path = "best.pt"
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    score_p1, score_p2 = 0, 0
    show_countdown = False
    countdown_start = 0
    countdown_duration = 5
    font, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0), 2
    game_over = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        mid_x = width // 2
        cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)

        # Oyun Bitti mi?
        if game_over:
            winner_text = "PLAYER 1 WINS!" if score_p1 == 3 else "PLAYER 2 WINS!"
            cv2.putText(frame, "CONGRATULATIONS!", 
                        (width//4, height//2 - 30), font, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, winner_text, 
                        (width//4, height//2 + 30), font, 1.5, (0, 255, 255), 3)
            cv2.imshow("Rock-Paper-Scissors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Geri Sayım
        if show_countdown:
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - int(elapsed)
            if remaining > 0:
                text = f"Next round in {remaining}"
                cv2.putText(frame, text, (width//2 - 100, height//2), font, 1.5, (0, 0, 255), 3)
                cv2.imshow("Rock-Paper-Scissors", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                show_countdown = False  # Countdown tamamlandı

        # Gesture Tanıma
        left_frame, right_frame = frame[:, :mid_x], frame[:, mid_x:]
        results_left = model.predict(left_frame, imgsz=224)
        results_right = model.predict(right_frame, imgsz=224)

        p1_gesture, p2_gesture = "no_gesture", "no_gesture"
        if results_left and len(results_left[0].probs) > 0:
            probs_np_left = results_left[0].probs.data.cpu().numpy()
            conf_left = np.max(probs_np_left)
            if conf_left > 0.85:
                top1_idx_left = int(np.argmax(probs_np_left))
                p1_gesture = model.names[top1_idx_left]

        if results_right and len(results_right[0].probs) > 0:
            probs_np_right = results_right[0].probs.data.cpu().numpy()
            conf_right = np.max(probs_np_right)
            if conf_right > 0.85:
                top1_idx_right = int(np.argmax(probs_np_right))
                p2_gesture = model.names[top1_idx_right]

        # Oyuncuların gesture'larını göster
        cv2.putText(frame, f"P1: {p1_gesture}", (10, 50), font, 1, color, thickness)
        cv2.putText(frame, f"P2: {p2_gesture}", (mid_x + 10, 50), font, 1, color, thickness)

        # Her iki oyuncunun geçerli gesture yapmasını bekle
        valid_gestures = ['rock', 'paper', 'scissors']
        if p1_gesture in valid_gestures and p2_gesture in valid_gestures:
            winner = get_winner(p1_gesture, p2_gesture)
            if winner == 'player1':
                score_p1 += 1
                show_countdown = True
                countdown_start = time.time()  # Countdown başlat
            elif winner == 'player2':
                score_p2 += 1
                show_countdown = True
                countdown_start = time.time()  # Countdown başlat

        # Skorları göster
        cv2.putText(frame, f"Score P1: {score_p1}", (10, height - 20), font, 1, (255, 255, 255), thickness)
        cv2.putText(frame, f"Score P2: {score_p2}", (mid_x + 10, height - 20), font, 1, (255, 255, 255), thickness)

        # Kazananı belirle
        if score_p1 == 5 or score_p2 == 5:
            game_over = True

        # Frame'i göster
        cv2.imshow("Rock-Paper-Scissors", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
