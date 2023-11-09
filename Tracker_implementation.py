import cv2
import numpy as np
import matplotlib.pyplot as plt
vid = cv2.VideoCapture("dharneesh_video.mp4")#capturing the video input
temp=cv2.imread("sat.png") #input refernce image
print(temp.size)
temp=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY) #convert video to gray

#old one.
# print(temp.shape)
#inital : 67,65
#58,46
temp=cv2.resize(temp,(58,46))

print(temp.shape)

# print(temp.shape)
prev_x = 0
prev_y = 0
x=0.00
count = 0
while(True):
    ret, frame = vid.read()  #read individual frames in video till video gets over
    print(frame.shape)

    #old frame
    #576,736.
    # frame=cv2.resize(frame,(256,256))
    count+=1
    frame=cv2.resize(frame,(512,512))
    if frame is None:
        break
    fr=frame
    fr=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    S=fr.shape
    F=temp.shape
    R=S[0]+F[0]-1
    C=S[1]+F[1]-1
    Z=np.zeros((R,C))
    for i in range(S[0]):   #zero padding of input frame ,no need to mention this
        for j in range(S[1]):
            Z[i+np.int32((F[0]-1)/2),j+np.int32((F[1]-1)/2)]=fr[i,j]

    p1=int(0)
    p2=int(0)
    mini=10000000000
    for i in range(200,S[0]-200):
        for j in range(200,S[1]-200):  #we iterate through the throughout the image
            k1=Z[i:i+F[0],j:j+F[1]]    #The refernce window with i,j as top left corner
            l=np.sum(np.multiply(k1,temp))
            m=np.sum(np.multiply(k1,k1))
            n=np.sum(np.multiply(temp,temp))
            if(m+n-2*l<mini):            #find the refernce window which has minimum error compared and corresponding top left corner
                p1=i
                p2=j
                mini=m+n-2*l
                gcur=k1


    #from here feature based tracker starts
    #if the minimum error is above certain limit we switch to this tracker
    if (mini > 100000 and count>1):
            sobelx = cv2.Sobel(fr,-100,1,0,ksize=3)#apply sobel operator in x,y directions  to give vertical,horizontal edges in th image
            sobely = cv2.Sobel(fr,-100,0,1,ksize=3)
            size = sobelx.shape
            container = sobelx

            for i in range(1, size[0]-1):# use a variable to store overall edge detections which includes x,y by taking square of individula sobel x,y and root it
                 for j in range(1, size[0]-1 ):
                     container[i][j] = np.sqrt(sobelx[i][j]**2 + sobely[i][j]**2)
            laplacian = cv2.Laplacian(fr,-1)# Laplacian filter is a second-order derivative filter, detect additional edges

            th, binarized = cv2.threshold(laplacian, 0 , 255, cv2.THRESH_OTSU)#apply otsu algorithm to binarize the image
            kernel = np.ones((3,3), np.uint8)
            eroded = cv2.erode(binarized, kernel, iterations=10)#apply erode to remove noise,dialate to connect 2 diffent shapes very near by
            dilated = cv2.dilate(binarized, kernel, iterations=1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)#returns the connected components in image

            # Print the coordinates of each object
            new_centriodsx  = []
            new_centriodsy = []
            for i in range(num_labels - 1):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx = int(centroids[i, 0])
                cy = int(centroids[i, 1])
                cv2.rectangle(dilated,(cx-10,cy-10),(cx+10,cy+10),(255,0,0),1)
                new_centriodsx.append(x)#appends coordinates of centeroids of each component
                new_centriodsy.append(y)
                print("Object {}: x={}, y={}, w={}, h={}, cx={}, cy={}".format(i+1, x, y, w, h, cx, cy))
            #till the above prescreener operation is being done
            mean_prev, variance_prev = cv2.meanStdDev(fr[int(prev_y)-23:int(prev_y)+23, int(prev_x)-29:int(prev_x)+29])#computes mean ,variance of previous template which computed in correlation based tracker
            print(mean_prev)
            print(type(mean_prev))
            M = 10000000000
            no_of_centr = len(new_centriodsx)
            for i in range(no_of_centr):#iterate through centroids and get to seatrch for  the best possible centriod of target
                if((int(new_centriodsy[i])-23)>=0 and  (int(new_centriodsx[i])-29)>=0 and (int(new_centriodsy[i])+23)<=512 and (int(new_centriodsx[i])+29)<=512):
                    pres_mean, pres_vari = cv2.meanStdDev(frame[int(new_centriodsy[i])-23:int(new_centriodsy[i])+23, int(new_centriodsx[i])-29:int(new_centriodsx[i])+29])
                    if ((abs(pres_mean[0][0]-mean_prev[0][0])+abs(variance_prev[0][0]-pres_vari[0][0])) < M):#if thhe mean,variance corresponding to segment is near to previous template then it is more probaby the target
                        M = abs(pres_mean[0][0]-mean_prev[0][0])+abs(variance_prev[0][0]-pres_vari[0][0])
                        p1 = prev_x
                        p2 = prev_y
    prev_x = p1
    prev_y = p2
    print(p1," ",p2,mini)
    if(x<0.05):
        x+=(1/1000)
    temp=np.add((1-x)*temp,x*gcur)#since the target image might chnage due to closer movement,hence we change the template image
    cv2.rectangle(frame,(p1-10,p2-10),(p1+10,p2+10),(255,0,0),1)
    cv2.imshow('frame', frame)#show the output which contains detected target



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()
