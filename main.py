import cv2
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card


def main():
    cap = cv2.VideoCapture(3)

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binary Result', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('Original', 600, 400)
    cv2.resizeWindow('Contours', 600, 400)
    cv2.resizeWindow('Binary Result', 400, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        card_found, corners, contour_preview, edges = detect_card(frame)

        cv2.imshow('Original', frame)
        cv2.imshow('Contours', contour_preview)

        if card_found and corners is not None:
            _, binary_warped = get_warped_card(frame, corners)
            cv2.imshow('Binary Result', binary_warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
