import numpy as np
import cv2
import json

# ============================================================================

FINAL_LINE_COLOR = (0, 255, 0)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name, pic):
        self.window_name = window_name # Name for our window
        self.polydone = False
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.Polypoints = [] # List of points defining our polygon
        self.img=pic
        
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.polydone: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.Polypoints), x, y))
            self.Polypoints.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click means deleting last point
            del self.Polypoints[-1]
            print("Removing point #%d" % (len(self.Polypoints)))
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # Left double click means we're done with polygon
            print("Completing polygon with %d points." % len(self.Polypoints))
            self.polydone = True
            


    def run(self):
        self.polydone = False
        self.Polypoints = []
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.img)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.polydone):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            if (len(self.Polypoints) > 1):
                # Draw all the current polygon segments
                cv2.polylines(self.img, np.array([self.Polypoints]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                # cv2.line(self.img, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, self.img)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            cv2.waitKey(50)

        # User finised entering the polygon points, so let's make the final drawing
        # of a filled polygon
        if (len(self.Polypoints) > 0):
            cv2.fillPoly(self.img, np.array([self.Polypoints]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, self.img)
        # Waiting for the user to press any key

        return self.img,  self.Polypoints

# ============================================================================

if __name__ == "__main__":

    polygons=[]
    pd = PolygonDrawer("Polygon", cv2.imread('1.jpg', 1))
    i=0
    try:
        while(True):
            image, p = pd.run()
            polygons.append(p)
            print("polygon #%d finished"%i)
            i+=1
    except KeyboardInterrupt:
        pass
    cv2.destroyWindow(pd.window_name)
    cv2.imwrite("polygon.png", image)
    with open('coords.json', 'w') as f:
        f.write(json.dumps(polygons))
    print("Polygon = %s" % str(polygons))
