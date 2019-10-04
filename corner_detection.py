import numpy as np
import cv2

def harris_detection(img,ksize=3):

    k=0.04
    threhold=0.05
    NMS=True

    h=img.shape[0]
    w=img.shape[1]

    grad=np.zeros((h,w,2),dtype=np.float32)
    grad[:,:,0]=cv2.Sobel(img,cv2.CV_16S,1,0,ksize=3)
    grad[:,:,1]=cv2.Sobel(img,cv2.CV_16S,0,1,ksize=3)
    # print("grad=",np.amax(grad))

    I=np.zeros((h,w,3),dtype=np.float32)
    I[:,:,0]=grad[:,:,0]**2
    I[:,:,1]=grad[:,:,1]**2
    I[:,:,2]=grad[:,:,0]*grad[:,:,1]
    # print("I=", np.amax(I))

    m=np.zeros((h,w,3),dtype=np.float32)
    m[:,:,0]=cv2.GaussianBlur(I[:,:,0],ksize=(ksize,ksize),sigmaX=2)
    m[:,:,1]=cv2.GaussianBlur(I[:,:,1],ksize=(ksize,ksize),sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(I[:, :, 2], ksize=(ksize, ksize), sigmaX=2)
    # print("m=", np.amax(m))
    m=[np.array([[m[i,j,0],m[i,j,2]],[m[i,j,2],m[i,j,1]]]) for i in range(h) for j in range(w)]

    D=list(map(np.linalg.det,m))
    T=list(map(np.trace,m))
    # for i in range(len(D)):
    #     print(i,D[i])
    R=np.array([d-k*t*t for d,t in zip(D,T)])

    R_max=max(R)
    # print('r_MAX=',R_max)
    R=R.reshape((h,w))
    corner=np.zeros_like(R,dtype=np.uint8)
    print("像素坐标：")
    for i in range(h):
        for j in range(w):

            if NMS == True:

                if R[i,j] > threhold*R_max and R[i,j]==np.max(R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]):
                    corner[i,j]=255
                    print(i,j)
                    # a=R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]
                    # print(a.shape[:2])
                    # print(np.max(R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]))
                    # print(R[i,j],"\n")

                    # corner[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]=255
                    # corner[i,j]=0
            else:
                if R[i,j] > threhold*R_max:
                    corner[i,j]=255
                    print(i, j)
                    # corner[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)] = 255
                    # corner[i, j] = 0
    return corner

def sub_pixel(dst,gray):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # print(centroids)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    print('亚像素坐标：')
    print(corners)
    res = np.hstack((centroids, corners))
    res = np.int0(res)

    return res


if __name__=='__main__':
    img=cv2.imread('/home/longtong/桌面/image/new_version/light_spot.bmp')
    # img=cv2.resize(img,(600,600))

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst=harris_detection(gray)

    res=sub_pixel(dst,gray)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]

    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
# import numpy as np
# import cv2
#
# img = cv2.imread('/home/longtong/桌面/image/new_version/corner_detection.bmp')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# # find Harris corners
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,3,3,0.04)
# # dst = cv2.dilate(dst,None)
# ret, dst = cv2.threshold(dst,0.05*dst.max(),255,0)
# dst = np.uint8(dst)
#
# # find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# print(centroids)
# # define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# print(corners)
# # Now draw them
# res = np.hstack((centroids,corners))
# res = np.int0(res)
# print(res)
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]] = [0,255,0]
#
#
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()