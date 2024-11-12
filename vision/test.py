import cv2

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect and compute keypoints and descriptors for the first frame
keypoints1, descriptors1 = orb.detectAndCompute(prev_gray, None)

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and compute keypoints and descriptors for the current frame
    keypoints2, descriptors2 = orb.detectAndCompute(gray, None)

    # Match descriptors using the BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the best matches (you can adjust the number)
    match_img = cv2.drawMatches(prev_gray, keypoints1, gray, keypoints2, matches[:10], None, flags=2)
    cv2.imshow('ORB Feature Matching', match_img)

    # Update previous frame and keypoints for the next iteration
    prev_gray = gray.copy()
    keypoints1, descriptors1 = keypoints2, descriptors2

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
