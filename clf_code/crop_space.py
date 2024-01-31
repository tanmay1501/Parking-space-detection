import os 

import cv2




def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):
        #area of the component:
        area =  values[i, cv2.CC_STAT_AREA]

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        
        slots.append([x1, y1, w, h])

    return slots


ouput_dir = './clf_data/all_'
mask = './data/mask_crop.png'


video_path = './data/parking_crop_loop.mp4'


mask = cv2.imread(mask,0)

cap = cv2.VideoCapture(video_path)


connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S )

spots = get_parking_spots_bboxes(connected_components)


ret = True

while ret:
    ret, frame = cap.read()
    for spot in spots:
        x1, y1, w, h = spot
        frame =  cv2.rectangle(frame, (x1, y1,), (x1+w,y1+h), (255,0,0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()