import matplotlib.pyplot as plt
import numpy as np
# ---
import cv2 as cv
# ---
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def png_2_RGB(file):
    """
    Convertit l'image 'file' au format png en 3 matrices 
    -------------
    paramètres :
    file : nom du fichier png à convertir 
    
    retourne :
    M_R, M_G, M_B : les matrices contenant l'intensité (sur 255) repectivement du rouge, vert et bleu en chaque point 
    """
    
    #plt.imread retourne l'image au bon format avec une quatrième valeur pour la transparence qu'on ne conserve pas 
    M_RGB  = plt.imread(file)[:,:,:3]
    #M_RGB contient des valeurs entre 0 et 1 qui correspondent en fait à des entiers entre 0 et 255
    M_RGB *= 255
    #passage en entier (le stockage des entier allant de 1 à 255 est plus faible que celui des flottants)
    M_RGB  = M_RGB.astype(np.int)
    
    return M_RGB[:,:,0], M_RGB[:,:,1], M_RGB[:,:,2]


# on utilisera dans un premier temps l'image suivante
M_R, M_G, M_B = png_2_RGB('ai_ready/images/silos_256-0-0--6-16--635-28790.png')

def image_diff(image):
    img_diff=image[:-1,:-1]-image[1:,1:]
    return img_diff

clipped = image_diff((M_R+M_G+M_B))+64
print(clipped)

plt.figure()
plt.imshow(clipped)
plt.savefig('teste.png')
plt.show()

# -----
image=clipped

distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.savefig('teste2.png')
plt.show()

# ---


img = cv.imread('opencv-logo-white.png',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()