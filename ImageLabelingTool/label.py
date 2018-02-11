import numpy as np
import cv2
import json

# ============================================================================

FINAL_LINE_COLOR = (0, 255, 0)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name, pic, polygons):
        self.window_name = window_name # Name for our window
        self.polydone = False
        self.dump = False
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.Polypoints = [] # List of points defining our polygon
        self.img=pic.copy()
        self.orig=pic.copy()
        self.polygons=polygons
        
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
            self.Polypoints = self.Polypoints[:-1]
            print("Removing point #%d" % (len(self.Polypoints)))
            #img=original for each polygone in polygons draw a polygon on img
            #tut ne pererisovivaet
            self.img=self.orig.copy()
            for poly in self.polygons:
                cv2.fillPoly(self.img, np.array([poly]), FINAL_LINE_COLOR) 
            cv2.polylines(self.img, np.array([self.Polypoints]), False, FINAL_LINE_COLOR, 1)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # Left double click means we're done with polygon
            print("Completing polygon with %d points." % len(self.Polypoints))
            self.polydone = True
            


    def drawPoly(self):
        self.polydone = False
        self.Polypoints = []
        # Let's create our working window and set a mouse callback to handle events
        GUI_NORMAL = 16 # workaround to properly disable a context menu on right button click
        cv2.namedWindow(self.window_name, flags=(cv2.WINDOW_NORMAL|GUI_NORMAL))
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

    
    def run(self):
        i=1
        while(not self.dump):
            self.drawPoly()
            self.polygons.append(self.Polypoints)
            print("polygon #%d finished"%i)
            i+=1
            confirm=False
            while (not confirm):
                rval=cv2.waitKey(0) #needs be optimazid maybe in multithreading
                if (rval==ord('c')):
                    print("Confirmed")
                    confirm=True
                elif (rval==ord('s')):
                    print("Saved")
                    self.dump=True
                    confirm=True
                elif(rval==ord('z')):
                    print("Canceled")
                    i-=1
                    #print("before")
                    #print(len(self.polygons))
                    self.polygons = self.polygons[:-1]
                    #print("after")
                    #print(len(self.polygons))
                    self.img=self.orig.copy()
                    for poly in self.polygons:
                        cv2.fillPoly(self.img, np.array([poly]), FINAL_LINE_COLOR)
                    cv2.imshow(self.window_name, self.img)
        return self.polygons

# ============================================================================


def typeOfWork():
    print("Type n for new work or c to continue work")
    while True:
        rval=input()
        if rval=='n':
            return True
        elif rval=='c':
            return False
        else:
            print("Invalid input.\nType n for new work or c to continue work")
    
if __name__ == "__main__":

    polygons=[]
    
if typeOfWork():
    #code for new work
    #for new picture we just take one argument - path to picture
    #parse path and pass it to imread method
    print("Print path to image file (includin file).\nEx.:/home/user/folder/image.jpg")
    pathToImg=input()
    #print(type(pathToImg))
else:
    #core for continueing work    
    #for continue work we ask to arguments - path to picture and path to json file
    #parse both paths, pass picture to imread method and draw all polygons
    print("Print path to image file (includin file).\nEx.:/home/user/folder/image.jpg")
    pathToImg=input()
    print("Print path to json file (includin file).\nEx.:/home/user/folder/labels.json")
    pathToJson=input()
    #print(pathToJson)
    polygons=json.load(open(pathToJson))
    print (polygons)
    
    
    
img=cv2.imread(pathToImg, 1)
pd = PolygonDrawer("Polygon", img, polygons)
polygons=pd.run()
    
cv2.destroyWindow(pd.window_name)
for poly in polygons:
    cv2.fillPoly(img, np.array([poly]), FINAL_LINE_COLOR)
cv2.imwrite("polygon.png", img)
with open('coords.json', 'w') as f:
    f.write(json.dumps(polygons))
print("Polygon = %s" % str(polygons))
