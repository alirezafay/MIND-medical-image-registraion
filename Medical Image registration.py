#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load images
mri_image = np.array(Image.open('MRI.png').convert('L'), dtype=np.float64)
pet_image = np.array(Image.open('PET.png').convert('L'), dtype=np.float64)


# In[3]:


def compute_D_p(image, x1, x2, P):
    D_p = 0
    for p in P:
        x1_p = (x1[0] + p[0], x1[1] + p[1])
        x2_p = (x2[0] + p[0], x2[1] + p[1])
        
        if (0 <= x1_p[0] < image.shape[0] and 0 <= x1_p[1] < image.shape[1] and
            0 <= x2_p[0] < image.shape[0] and 0 <= x2_p[1] < image.shape[1]):
            D_p += (image[x1_p[0], x1_p[1]] - image[x2_p[0], x2_p[1]]) ** 2
    return D_p

def compute_V(image, x, N, P):
    V = 0
    for n in N:
        x_n = (x[0] + n[0], x[1] + n[1])
        
        if 0 <= x_n[0] < image.shape[0] and 0 <= x_n[1] < image.shape[1]:
            V += compute_D_p(image, x, x_n, P)
    V /= len(N)
    return V

def mind_descriptor(image, x, R, P, N):
    V_x = compute_V(image, x, N, P)
    epsilon = 1e-8  
    mind = np.zeros(len(R))  
    for i, r in enumerate(R):
        x_r = (x[0] + r[0], x[1] + r[1])
        
        if 0 <= x_r[0] < image.shape[0] and 0 <= x_r[1] < image.shape[1]:
            D_p_value = compute_D_p(image, x, x_r, P)
            mind_value = np.exp(-D_p_value / (V_x + epsilon))  
            mind[i] = mind_value
    if np.max(mind) != 0:  
        mind /= np.max(mind) 
    return mind

def compute_mind_descriptors(image, R, P, N):
    descriptors = np.zeros((image.shape[0], image.shape[1], len(R)))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            descriptors[i, j] = mind_descriptor(image, (i, j), R, P, N)
    return descriptors


# In[4]:


P = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
N = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
R = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  

mri_mind_descriptors = compute_mind_descriptors(mri_image, R, P, N)
pet_mind_descriptors = compute_mind_descriptors(pet_image, R, P, N)


# In[5]:


# select a point in MRI image
mri_image_point = mri_mind_descriptors[100,100]

height = mri_mind_descriptors.shape[0]
width = mri_mind_descriptors.shape[1]

similarity_map = np.zeros([height,width])

for h in range(height):
    for w in range(width):
        similarity_map[h,w] = np.mean((mri_image_point - pet_mind_descriptors[h,w])**2)

# # Compute similarity map (mean squared difference of descriptors)
# similarity_map = np.mean((mri_mind_descriptors - pet_mind_descriptors) ** 2, axis=2)

plt.figure(figsize=(7, 7))
plt.title('MIND Similarity Heatmap')
plt.imshow(similarity_map, cmap='hot')
plt.colorbar()
plt.show()


# In[6]:


import random

random.seed(10)
num_points = 20
height, width = mri_image.shape
random_points = [(random.randint(0, height-1), random.randint(0, width-1)) for _ in range(num_points)]

plt.figure(figsize=(5, 5))
plt.imshow(mri_image, cmap='gray')
for point in random_points:
    plt.plot(point[1], point[0], 'ro')
plt.title('Selected Points on MRI Image')
plt.show()


# In[7]:


import cv2

angle = 30
M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
rotated_pet_image = cv2.warpAffine(pet_image, M, (width, height))

# Display the rotated PET image
plt.figure(figsize=(5, 5))
plt.imshow(rotated_pet_image, cmap='gray')
plt.title('Rotated PET Image')
plt.show()


# In[8]:


def find_corresponding_point_local(mind_descriptor, target_descriptors, point, search_radius, R, P, N):
    min_distance = float('inf')
    corresponding_point = None
    x, y = point
    
    # Define the search window
    for i in range(max(0, x - search_radius), min(target_descriptors.shape[0], x + search_radius + 1)):
        for j in range(max(0, y - search_radius), min(target_descriptors.shape[1], y + search_radius + 1)):
            distance = np.mean((mind_descriptor - target_descriptors[i, j]) ** 2)
            if distance < min_distance:
                min_distance = distance
                corresponding_point = (i, j)
    
    return corresponding_point

search_radius = 20

rotated_pet_mind_descriptors = compute_mind_descriptors(rotated_pet_image, R, P, N)
corresponding_points = []
for point in random_points:
    mri_mind = mind_descriptor(mri_image, point, R, P, N)
    corresponding_point = find_corresponding_point_local(mri_mind, rotated_pet_mind_descriptors, point, search_radius, R, P, N)
    corresponding_points.append(corresponding_point)

# Display corresponding points on the rotated PET image
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(rotated_pet_image, cmap='gray')
for point in corresponding_points:
    plt.plot(point[1], point[0], 'ro')
plt.title('Corresponding Points on Rotated PET Image')
plt.subplot(1,2,2)
plt.imshow(mri_image, cmap='gray')
for point in random_points:
    plt.plot(point[1], point[0], 'ro')
plt.title('Selected Points on MRI Image')
plt.show()


# In[9]:


src_points = np.array(random_points)
dst_points = np.array(corresponding_points)

A = np.vstack([src_points.T, np.ones(len(src_points))]).T
B = np.vstack([dst_points.T, np.ones(len(dst_points))]).T

rotation_matrix, res, rank, s = np.linalg.lstsq(A, B, rcond=None)

calculated_rotation_matrix = rotation_matrix[:2, :2]

theta = np.radians(30)
known_rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
error = np.linalg.norm(calculated_rotation_matrix - known_rotation_matrix)

print("Calculated Rotation Matrix:")
print(calculated_rotation_matrix)
print("\nKnown Rotation Matrix:")
print(known_rotation_matrix)
print("\nError:")
print(error)


# In[10]:


angles = [10, 30, 90, 180]

results = []

for angle in angles:
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_pet_image = cv2.warpAffine(pet_image, M, (width, height))

    rotated_pet_mind_descriptors = compute_mind_descriptors(rotated_pet_image, R, P, N)
    corresponding_points = []
    for point in random_points:
        mri_mind = mind_descriptor(mri_image, point, R, P, N)
        corresponding_point = find_corresponding_point_local(mri_mind, rotated_pet_mind_descriptors, point, search_radius, R, P, N)
        corresponding_points.append(corresponding_point)

    src_points = np.array(random_points)
    dst_points = np.array(corresponding_points)

    A = np.vstack([src_points.T, np.ones(len(src_points))]).T
    B = np.vstack([dst_points.T, np.ones(len(dst_points))]).T
    
    rotation_matrix, res, rank, s = np.linalg.lstsq(A, B, rcond=None)
    calculated_rotation_matrix = rotation_matrix[:2, :2]

    theta = np.radians(angle)
    known_rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    error = np.linalg.norm(calculated_rotation_matrix - known_rotation_matrix)
    results.append((angle, calculated_rotation_matrix, known_rotation_matrix, error))

    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(rotated_pet_image, cmap='gray')
    for point in corresponding_points:
        plt.plot(point[1], point[0], 'ro')
    plt.title('Corresponding Points on Rotated PET Image')
    plt.subplot(1,2,2)
    plt.imshow(mri_image, cmap='gray')
    for point in random_points:
        plt.plot(point[1], point[0], 'ro')
    plt.title('Selected Points on MRI Image')
    plt.show()


# Print results
for angle, calc_mat, known_mat, error in results:
    print(f"Angle: {angle} degrees")
    print("Calculated Rotation Matrix:")
    print(calc_mat)
    print("Known Rotation Matrix:")
    print(known_mat)
    print("Error:")
    print(error)
    print()


# In[11]:


def add_gaussian_noise(image, mean=0, var=1000):
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    num_pepper = np.ceil(pepper_prob * total_pixels)

    # Add salt noise
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


# In[12]:


# Add noise to the images
noisy_mri_image = add_gaussian_noise(mri_image)
noisy_mri_image = add_salt_and_pepper_noise(noisy_mri_image)
noisy_pet_image = add_gaussian_noise(pet_image)
noisy_pet_image = add_salt_and_pepper_noise(noisy_pet_image)

# Display the noisy images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Noisy MRI Image')
plt.imshow(noisy_mri_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Noisy PET Image')
plt.imshow(noisy_pet_image, cmap='gray')
plt.show()


# In[13]:


mri_mind_descriptors = compute_mind_descriptors(noisy_mri_image, R, P, N)

random.seed(10)
num_points = 20
height, width = noisy_mri_image.shape
random_points = [(random.randint(0, height-1), random.randint(0, width-1)) for _ in range(num_points)]

plt.figure(figsize=(5, 5))
plt.imshow(noisy_mri_image, cmap='gray')
for point in random_points:
    plt.plot(point[1], point[0], 'ro')
plt.title('Selected Points on Noisy MRI Image')
plt.show()


# In[14]:


search_radius = 20

angles = [10, 30, 90, 180]

results = []

for angle in angles:
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_pet_image = cv2.warpAffine(noisy_pet_image, M, (width, height))

    rotated_pet_mind_descriptors = compute_mind_descriptors(rotated_pet_image, R, P, N)
    
    corresponding_points = []
    for point in random_points:
        mri_mind = mind_descriptor(noisy_mri_image, point, R, P, N)
        corresponding_point = find_corresponding_point_local(mri_mind, rotated_pet_mind_descriptors, point, search_radius, R, P, N)
        corresponding_points.append(corresponding_point)

    src_points = np.array(random_points)
    dst_points = np.array(corresponding_points)

    A = np.vstack([src_points.T, np.ones(len(src_points))]).T
    B = np.vstack([dst_points.T, np.ones(len(dst_points))]).T
    
    rotation_matrix, res, rank, s = np.linalg.lstsq(A, B, rcond=None)
    calculated_rotation_matrix = rotation_matrix[:2, :2]

    theta = np.radians(angle)
    known_rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    error = np.linalg.norm(calculated_rotation_matrix - known_rotation_matrix)
    results.append((angle, calculated_rotation_matrix, known_rotation_matrix, error))

    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(rotated_pet_image, cmap='gray')
    for point in corresponding_points:
        plt.plot(point[1], point[0], 'ro')
    plt.title(f'Corresponding Points on Rotated Noisy PET Image ({angle} degrees)')
    plt.subplot(1,2,2)
    plt.imshow(noisy_mri_image, cmap='gray')
    for point in random_points:
        plt.plot(point[1], point[0], 'ro')
    plt.title('Selected Points on MRI Image')

    plt.show()

# Print results
for angle, calc_mat, known_mat, error in results:
    print(f"Angle: {angle} degrees")
    print("Calculated Rotation Matrix:")
    print(calc_mat)
    print("Known Rotation Matrix:")
    print(known_mat)
    print("Error:")
    print(error)
    print()

