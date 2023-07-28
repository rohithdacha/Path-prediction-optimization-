from random import randint
import time
import cv2
import numpy as np
from collections import defaultdict

class MyVehicle:
    tracks = []
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
    def getRGB(self):
        return (self.R,self.G,self.B)
    def getTracks(self):
        return self.tracks
    def getId(self):
        return self.i
    def getState(self):
        return self.state
    def getDir(self):
        return self.dir
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done
    def going_UP(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end: #Check if Upper line is crossed
                    state = '1'
                    self.dir = 'up'
                    return True
            else:
                return False
        else:
            return False
    def going_DOWN(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start: #Check if Lower line is crossed
                    state = '1'
                    self.dir = 'down'
                    return True
            else:
                return False
        else:
            return False
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
class MultiPerson:
    def __init__(self, persons, xi, yi):
        self.vehicles = vehicles
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False

cnt = 1
areac = 1
areaTH = 1

def Count(link):
    cnt_up = 0
    cnt_down = 0
    cnt_right=0

    cap = cv2.VideoCapture(link) # read the video

    for i in range(19):
        (i, cap.get(i))

    w = cap.get(3)
    h = cap.get(4)
    frameArea = h*w
    areaTH = frameArea/200
    print('Area Threshold', areaTH)
    
    # Upper/Lower Lines
    line_up = int(2*(h/6))
    line_down = int(3*(h/5))

    up_limit = int(1*(h/6))
    down_limit = int(4*(h/5))

    print("Blue line y:", str(line_down))
    print("Red line y:", str(line_up))
    line_down_color = (255,0,0)
    line_up_color = (0,0,255)
    pt1 = [0, line_down]
    pt2 = [w, line_down]
    pts_L1 = np.array([pt1,pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1,1,2))
    pt3 = [0, line_up]
    pt4 = [w, line_up]
    pts_L2 = np.array([pt3,pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1,1,2))

    pt5 = [0, up_limit]
    pt6 = [w, up_limit]
    pts_L3 = np.array([pt5,pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1,1,2))
    pt7 = [0, down_limit]
    pt8 = [w, down_limit]
    pts_L4 = np.array([pt7,pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1,1,2))

    #Create the background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    kernelOp = np.ones((3,3), np.uint8)
    kernelOp2 = np.ones((5,5), np.uint8)
    kernelCl = np.ones((11,11), np.uint8)

    #Variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    vehicles = []
    max_p_age = 5
    pid = 1

    while(cap.isOpened()):
        #read a frame
        ret, frame = cap.read()

        for i in vehicles:
            i.age_one()     # age every Car on frame

        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg.apply(frame)

        # Binary to remove shadow
        try:
            #cv2.imshow('Frame', frame)
            #cv2.imshow('Backgroud Subtraction', fgmask)
            ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)
            #Opening (erode->dilate) to remove noise
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
            #Closing (dilate->erode) to join white region
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
            #cv2.imshow('Image Threshold', cv2.resize(fgmask, (400, 300)))       #frame size= 400*300
            #cv2.imshow('Image Threshold2', cv2.resize(fgmask2, (400, 300)))		#frame size= 400*300
            #cv2.imshow('Masked Image', cv2.resize(mask, (400, 300)))			#frame size= 400*300
            #cv2.imshow('Masked Image2', cv2.resize(mask2, (400, 300)))			#frame size= 400*300
        except:
            
            print('End Of Frames')
            #print('Up:', cnt_up)
            cnt_right=cnt_down-cnt_up
            #print('Right:', cnt_right)
            break

        

        contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours0:
            cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
            area = cv2.contourArea(cnt)
            #print area," ",areaTH
        
            if area > areaTH and area < 20000:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)

                # the object is near the one which already detect before
                new = True
                for i in vehicles:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)   # Update the coordinates in the object and reset age
                        if i.going_UP(line_down, line_up) == True:
                            cnt_up += 1
                            #print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                        elif i.going_DOWN(line_down, line_up) == True:
                            roi = frame[y:y+h, x:x+w]
                            #cv2.imshow('Region of Interest', roi)
                            #print("Area equal to ::::", area)
                            cnt_down += 1
                            #print("ID:", i.getId(), 'crossed going down at', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        # Remove from the list of Vehicle
                        index = vehicles.index(i)
                        vehicles.pop(index)
                        del i
                if new == True:
                    p = MyVehicle(pid,cx,cy, max_p_age)
                    vehicles.append(p)
                    pid += 1

        
                cv2.circle(frame,(cx,cy),5, (0,0,255), -1)
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        for i in vehicles:
            cv2.putText(frame, str(i.getId()), (i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)

        
        str_up = 'UP:' + str(cnt_up)
        str_down = 'DOWN:' + str(cnt_down)
        frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=1)
        frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=1)
        frame = cv2.polylines(frame, [pts_L3], False, (255,255,255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255,255,255), thickness=1)
        #cv2.putText(frame, str_up, (5,45),font,2,(255,255,255),2,cv2.LINE_AA)
        #cv2.putText(frame, str_up, (5,45),font,2,(0,0,255),1,cv2.LINE_AA)
        #cv2.putText(frame, str_down, (5,95),font,2,(255,255,255),2,cv2.LINE_AA)
        #cv2.putText(frame, str_down, (5,95),font,2,(255,0,0),1,cv2.LINE_AA)


        #cv2.imshow('Backgroud Subtraction', fgmask)
        cv2.imshow('Frame', cv2.resize(frame, (400, 300)))
        

        #Abort and exit with ESC
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('COUNT: ',cnt_right)
    return cnt_right


link=[r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video1.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video2.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video3.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video4.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video5.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video6.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video7.mp4',
r'C:\Users\dacha rohith\OneDrive - vit.ac.in\Desktop\Winter semester 22-23\SDN project\project\Video8.mp4',
]

Congvalue = (cnt * areac)/ areaTH

edges = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('C', 'E'),
    ('B', 'D'),
    ('E', 'D'),
    ('D', 'F'),
    ('E', 'F')
]

for i in range(len(edges)):
    n=Count(link[i])
    edges[i] = list(edges[i])
    edges[i].insert(2,n)
    edges[i] = tuple(edges[i])


class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight
    
graph = Graph()

for edge in edges:
    graph.add_edge(*edge)
    # print(edge)

def dijsktra(graph, initial, end):
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

user_input1 = input('Enter the initial position :\n')
user_input2 = input('Enter the final position :\n')

print(dijsktra(graph, user_input1, user_input2))
