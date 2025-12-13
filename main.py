import numpy as np
import ultralytics
import cv2


model = ultralytics.YOLO('brain-tumor-seg.pt')
orig_img = cv2.imread('tumor1.jpg')

model_results = model.predict(orig_img)
orig_model_result = model_results[0]
orig_model_result_plot = orig_model_result.plot()

orig_model_result_mask = orig_model_result.masks.data[0]
orig_model_result_mask = orig_model_result_mask.numpy()

orig_model_result_mask_area_px = orig_model_result_mask.sum()
orig_model_result_mask_area_sm = orig_model_result_mask_area_px * 0.0025
print(orig_model_result_mask_area_sm)

if orig_model_result_mask_area_sm < 10:
    tumor_type = "small tumor"

elif 25 >= orig_model_result_mask_area_sm >= 10:
    tumor_type = "middle tumor"

else:
    tumor_type = "large tumor"

orig_model_result_mask = orig_model_result_mask.astype(np.uint8)
orig_model_result_mask *= 255
orig_model_result_mask = orig_model_result_mask.astype(bool)

orig_img[~orig_model_result_mask] = 0

cv2.imshow(f'{tumor_type}', orig_img)
cv2.waitKey(0)