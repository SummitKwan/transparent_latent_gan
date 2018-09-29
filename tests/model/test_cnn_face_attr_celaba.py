import importlib
import src.model.cnn_face_attr_celeba as cnn_face
importlib.reload(cnn_face)

img_name, df_attr = cnn_face.get_data_info()

model = cnn_face.create_cnn_model()

x, y = cnn_face.get_data_sample(yn_interactive_plot=True)

x_all, y_all = cnn_face.load_data_batch(num_images_total=2**16)

cnn_face.train_protocol()

model = cnn_face.create_cnn_model()
model.load_weights(cnn_face.get_list_model_save()[-1])


