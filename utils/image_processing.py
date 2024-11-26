import cv2
import numpy as np


def order_points(pts):
    """
    Mengurutkan empat titik yang merepresentasikan sudut-sudut kartu.
    Urutan: [top-left, top-right, bottom-right, bottom-left]
    """
    # Inisialisasi array untuk menyimpan urutan sudut
    rect = np.zeros((4, 2), dtype="float32")

    # Menentukan top-left dan bottom-right berdasarkan jumlah koordinat x + y
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left memiliki jumlah terkecil
    rect[2] = pts[np.argmax(s)]  # Bottom-right memiliki jumlah terbesar

    # Menentukan top-right dan bottom-left berdasarkan perbedaan koordinat x - y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right memiliki perbedaan terkecil
    rect[3] = pts[np.argmax(diff)]  # Bottom-left memiliki perbedaan terbesar

    return rect


def auto_rotate_points(points):
    """
    Mengatur ulang titik-titik agar kartu ditampilkan dengan orientasi yang benar.
    Jika kartu lebih lebar dari tinggi, titik-titik dirotasi 90 derajat.
    """
    # Memanggil order_points untuk mengurutkan titik terlebih dahulu
    ordered = order_points(points)

    # Menghitung tinggi kartu di sisi kiri dan kanan
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height = max(height_left, height_right)

    # Menghitung lebar kartu di bagian atas dan bawah
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    width = max(width_top, width_bottom)

    # Jika kartu lebih lebar dari tinggi, rotasi titik-titik 90 derajat
    if width > height:
        new_ordered = np.array([
            ordered[3],  # Bottom-left menjadi top-left
            ordered[0],  # Top-left menjadi top-right
            ordered[1],  # Top-right menjadi bottom-right
            ordered[2]   # Bottom-right menjadi bottom-left
        ])
        return new_ordered

    # Jika tinggi lebih besar, gunakan urutan asli
    return ordered


def get_warped_card(frame, corners, width=500, height=700):
    """
    Melakukan transformasi perspektif untuk meluruskan gambar kartu.
    """
    # Urutkan dan rotasi titik-titik sudut sesuai orientasi yang benar
    ordered_corners = auto_rotate_points(corners)

    # Menghitung lebar dan tinggi saat ini dari kartu yang terdeteksi
    current_width = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    current_height = np.linalg.norm(ordered_corners[3] - ordered_corners[0])

    # Tukar dimensi width dan height jika kartu lebih lebar dari tinggi
    if current_width > current_height:
        width, height = height, width

    # Tentukan koordinat titik tujuan (0,0) ke (width, height) pada gambar hasil
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Mendapatkan matriks transformasi perspektif
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    # Melakukan warp perspektif pada frame asli
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    # Konversi gambar hasil warp menjadi grayscale
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Konversi gambar grayscale menjadi biner dengan threshold adaptif
    binary_warped = cv2.adaptiveThreshold(
        gray_warped,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Memastikan background hitam dan kartu putih
    # Jika bagian atas gambar rata-rata terlalu terang, invert gambar
    if np.mean(binary_warped[0:50, 0:50]) > 127:
        binary_warped = cv2.bitwise_not(binary_warped)

    # Balik gambar biner lagi agar kartu berwarna putih di atas background hitam
    binary_warped = cv2.bitwise_not(binary_warped)

    # Mengembalikan gambar yang telah di-warp dan hasil gambar biner
    return warped, binary_warped
