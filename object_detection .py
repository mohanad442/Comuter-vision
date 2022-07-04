import cv2
import matplotlib.pyplot as plt
# %matplotlib inline


# In[5]:


config_file = "Desktop/object detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "Desktop/object detection/frozen_inference_graph.pb"


# In[6]:


# Using ssd_mobilenet_v3 model

model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[7]:


model.setInputSize(320,320)
model.setInputScale(1/127.5)   # 255 / 2 = 127.5
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# In[8]:


labels = []
file_name = "Desktop/object detection/labels.txt"

with open(file_name, "rt") as file:
  labels = file.read().rstrip('\n').split("\n")


# In[9]:


print("The length of labels is ", len(labels))
labels[:5]    # first 5 labels


# In[14]:


img1 = cv2.imread("Desktop/object detection/image.png")
img2 = cv2.imread("Desktop/object detection/image 2.jpeg")
img3 = cv2.imread("Desktop/object detection/img 3.jpg")


# In[8]:


plt.imshow(img1)
plt.figure()
plt.imshow(img2)
plt.figure()
plt.imshow(img3)


# In[15]:


# the difference between BGR and RGB
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))


# In[11]:


model.detect(img3, confThreshold = 0.5)

# classes [1, 3]
# Probability for each class [0.7962084, 0.5984135]
# Box edges for each class   [[409,  97, 231, 570],[155, 293, 839, 363]]


# In[12]:


font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

for img in [img1, img2, img3]:
    
  class_label, probabilty, boxes = model.detect(img, confThreshold = 0.5)  

  for label, prob, box in zip(class_label, probabilty,boxes):
        
    text = labels[label-1]    # class_label starts from 1 but labels index start from 0
    cv2.rectangle(img, box, (255,0,0), 3)
    cv2.putText(img, text,( box[0]+10 , box[1] + 40) , font , font_scale, color = (0,0,255) , thickness  = 2)
     
  plt.figure()
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    


# In[20]:


# Object detection on video

video = cv2.VideoCapture("Desktop/object detection/Cairokee.mp4")
if not video.isOpened():
  raise print("Cannot open the video")

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

while True:
    
  # Here we cuttoff the video into separated images  
    
  ret, img = video.read()
  # ret is a boolen output gets 1 if there's an image and 0 when the video ended (no more images)
  if ret:
      # this condition is important because when the video end there is nomore images and 
      # the img shape will be zero so it will get an error in cv2.putText
      class_label, probabilty, boxes = model.detect(img, confThreshold = 0.56)

      if (len(class_label) != 0):
        
        for label, prob, box in zip(class_label, probabilty,boxes):
            
          if label <= 80:  
          # sometimes it detect the label more than 80 
              text = labels[label-1]  
              cv2.rectangle(img, box, (255,0,0), 3)
              cv2.putText(img, text,( box[0]+10 , box[1] + 40),font, font_scale,color = (0,0,255),thickness= 2)
            
      cv2.imshow("Video" , img)
    
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# In[21]:


# Getting video from webcam

video = cv2.VideoCapture(0)
if not video.isOpened():
  raise print("Cannot open the video")

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

while True:

  ret, img = video.read()

  class_label, probabilty, boxes = model.detect(img, confThreshold = 0.6)

  if (len(class_label) != 0):
    
    for label, prob, box in zip(class_label, probabilty,boxes):
        
        if label <= 80 :
            
          text = labels[label-1]  
          cv2.rectangle(img, box, (255,0,0), 3)
          cv2.putText(img, text,( box[0]+10 , box[1] + 40) , font , font_scale, color = (0,0,255) , thickness  = 2)
        
  cv2.imshow("Video" , img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
