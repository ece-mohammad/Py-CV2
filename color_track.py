import cv2
import numpy as np
from collections import deque as queue

DIRECTIONS = {
    'stop':     0,    # stop, don't move
    'right':    1,    # move right
    'left':     2,    # move left
    'fwd':      3,    # move forward
    'bwd':      4,    # move back
}

REGIONS = {
    'forbidden':0,
    'right':    1,
    'left':     2,
    'top':      3,
    'bottom':   4,
}


def nothing(x):
    pass


def get_direction(old_point, new_point):

    ret = DIRECTIONS['stop']

    if (new_point[0] - old_point[0]) > 0:
        ret = DIRECTIONS['right']

    elif (new_point[0] - old_point[0]) < 0:
        ret = DIRECTIONS['left']

    elif (new_point[1] - old_point[1]) < 0:
        ret = DIRECTIONS['fwd']

    elif (new_point[1] - old_point[1]) > 0:
        ret = DIRECTIONS['bwd']

    else:
        ret = DIRECTIONS['stop']

    return ret


# img_size(height -y-, width -x-), point(x, y)
def get_region(point, img_size, debug=False):

    ret = REGIONS['forbidden']

    # point co-ords
    x_pos = point[0]
    y_pos = point[1]

    # image dimensions
    height = img_size[0]
    width = img_size[1]

    # img slices
    # divide width and height into 3 equal ranges
    x_slice = int(width/3)
    y_slice = int(height/3)

    # x_slices >---> vertical slices
    v_slices = [
            None,
            range(0, x_slice),
            range(x_slice, 2*x_slice),
            range(2*x_slice, width)
    ]

    # y_slices >---> horizontal slices
    h_slices = [
            None,
            range(0, y_slice),
            range(y_slice, 2 * y_slice),
            range(2 * y_slice, height)
    ]

    # TODO: calculate point region
    if v_slices[2].__contains__(x_pos) and h_slices[1].__contains__(y_pos):     # top
        ret = REGIONS['top']

    elif v_slices[2].__contains__(x_pos) and h_slices[3].__contains__(y_pos):   # bottom
        ret = REGIONS['bottom']

    elif v_slices[1].__contains__(x_pos) and h_slices[2].__contains__(y_pos):   # left
        ret = REGIONS['left']

    elif v_slices[3].__contains__(x_pos) and h_slices[2].__contains__(y_pos):   # right
        ret = REGIONS['right']

    else:
        ret = REGIONS['forbidden']

    if debug:
        print("Region :", {v:k for (k,v) in REGIONS.items()}[ret])

    return ret


def color_track():

    # capture cam feed
    cap = cv2.VideoCapture(0)

    trace_queue = queue()

    cv2.namedWindow("Color Controls")

    # pick tuning mask parameters
    # choose color
    cv2.createTrackbar("Hue", "Color Controls", 0, 179, nothing)
    cv2.createTrackbar("Sat", "Color Controls", 0, 255, nothing)
    cv2.createTrackbar("Val", "Color Controls", 0, 255, nothing)

    # tune delta
    cv2.createTrackbar("Hue Delta", "Color Controls", 0, 179, nothing)
    cv2.createTrackbar("Sat Delta", "Color Controls", 0, 255, nothing)
    cv2.createTrackbar("Val Delta", "Color Controls", 0, 255, nothing)

    # tune image effects
    cv2.createTrackbar("Blur", "Cam Feed", 1, 10, nothing)
    cv2.createTrackbar("Min Detection Area", "Color Controls", 100, 5000, nothing)

    # start event loop
    while True:

        # read frame from cam feed
        _, captured = cap.read()

        # blur the captured frame
        frame = cv2.blur(captured, (3, 3))

        # divide the screen into partitions
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]



        # transform frames to a suitable color format
        # Hue : color (tone) normal scale: [0: 180]
        # Saturation : color concentration [0: 255]
        # Value : color brightness [0: 255]
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # set mask bounds
        lower_bound = np.array([
            abs(cv2.getTrackbarPos("Hue", "Color Controls") - cv2.getTrackbarPos("Hue Delta", "Color Controls")),
            abs(cv2.getTrackbarPos("Sat", "Color Controls") - cv2.getTrackbarPos("Sat Delta", "Color Controls")),
            abs(cv2.getTrackbarPos("Val", "Color Controls") - cv2.getTrackbarPos("Val Delta", "Color Controls"))
        ], int)
        upper_bound = np.array([
            (cv2.getTrackbarPos("Hue", "Color Controls") + cv2.getTrackbarPos("Hue Delta", "Color Controls")) % 180,
            (cv2.getTrackbarPos("Sat", "Color Controls") + cv2.getTrackbarPos("Sat Delta", "Color Controls")) % 256,
            (cv2.getTrackbarPos("Val", "Color Controls") + cv2.getTrackbarPos("Val Delta", "Color Controls")) % 256
        ], int)

        # print("upper bound:", upper_bound)
        # print("lower bound:", *lower_bound)

        # TODO uncomment and remove hard coded color value
        # generate mask
        mask = cv2.inRange(processed_frame, np.array([0, 80, 109], int), np.array([178, 244, 251], int))
        # mask = cv2.inRange(processed_frame, lower_bound, upper_bound)

        # get contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if contours list is not empty
        if contours:
            # get largest detected contour
            largest_contour = max(contours, key=cv2.contourArea)

            # get contour's minimum enclosing circle
            ((cnt_cx, cnt_cy), cnt_r) = cv2.minEnclosingCircle(largest_contour)

            if cv2.contourArea(largest_contour) >= cv2.getTrackbarPos("Min Detection Area", "Color Controls"):
                # get contour's info centeroid
                cnt_moments = cv2.moments(largest_contour)
                cnt_center = int(cnt_moments['m10']/cnt_moments['m00']), int(cnt_moments['m01']/cnt_moments['m00'])
                cv2.drawContours(captured, largest_contour, -1, (0, 0, 255), 3)
                cv2.circle(captured, (int(cnt_cx), int(cnt_cy)), int(cnt_r), (0, 255, 0), 5)

                # update contour trace
                trace_queue.append(cnt_center)

        # TODO not sure if this works!!!
        else:
            trace_queue.clear()

        # draw trace
        for i in range(len(trace_queue) - 1):
            if trace_queue[i] and trace_queue[i+1]:
                cv2.line(captured, trace_queue[i], trace_queue[i+1], (255, 0, 0), 2)

        # draw areas
        width = captured.shape[1]
        height = captured.shape[0]
        cv2.rectangle(captured, (0, 0), (width, int(height/3)), (0, 255, 255), 3)
        cv2.rectangle(captured, (0, 0), (int(width/3), height), (0, 255, 255), 3)
        cv2.rectangle(captured, (int(2*(width/3)), 0), (width, height), (0, 255, 255), 3)
        cv2.rectangle(captured, (0, int(2*(height/3))), (width, height), (0, 255, 255), 3)

        # display captured frame
        cv2.imshow("Cam Feed", captured)
        cv2.imshow("Processed Feed", mask)

        # wait for key
        key = cv2.waitKey(30)

        # if key is ESC, exit program
        if key == 27:
            break
        # if key is Backspace, clear tracking points
        elif key == 8:
            trace_queue.clear()

        # return contour center
        yield get_region(cnt_center, captured.shape, debug=True)

    # release cam 0 from capture
    cap.release()

    # destroy all cv2 windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # blue : low [50, 75, 0] - high : [180, 170, 255]

    # reg = get_region((0, 0), (100, 100), debug=True)
    # print(0, 0, reg)
    # reg = get_region((45, 10), (100, 100), debug=True)
    # print(45, 10, reg)
    # reg = get_region((80, 10), (100, 100), debug=True)
    # print(80, 10, reg)
    # reg = get_region((13, 45), (100, 100), debug=True)
    # print(13, 45, reg)
    # reg = get_region((50, 45), (100, 100), debug=True)
    # print(50, 45, reg)
    # reg = get_region((80, 45), (100, 100), debug=True)
    # print(80, 45, reg)
    # reg = get_region((10, 80), (100, 100), debug=True)
    # print(10, 80, reg)
    # reg = get_region((45, 80), (100, 100), debug=True)
    # print(45, 80, reg)
    # reg = get_region((80, 80), (100, 100), debug=True)
    # print(80, 80, reg)

    for i in color_track():
        print("region: ", i)


