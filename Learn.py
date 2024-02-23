import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

base_dir = os.getcwd()
png_dir = os.path.join(base_dir, 'png')
img_height, img_width = 70, 70

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (4, 4), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (5, 5), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(250, activation='relu'),
        Dropout(0.4),
        Dense(250, activation='softmax')
    ])
    return model

model_25BS = create_model()
#model_32BS = create_model()

models = [model_25BS]
batch_sizes = [25]
accuracies = []

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=50,  # rozsah, ve kterém náhodně otočit obrázky
    width_shift_range=0.1,  # rozsah, ve kterém náhodně posunout obrázky horizontálně
    height_shift_range=0.3,  # rozsah, ve kterém náhodně posunout obrázky vertikálně
    shear_range=0.3,  # rozsah náhodného zkosení obrázků
    zoom_range=0.3  # rozsah náhodného přiblížení obrázků
)

validation_generator = train_datagen.flow_from_directory(
    png_dir,
    target_size=(img_height, img_width),
    batch_size=30,
    class_mode='categorical',
    subset='validation')

for model, batch_size in zip(models, batch_sizes):
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_generator = train_datagen.flow_from_directory(
        png_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    
    history = model.fit(train_generator, epochs=70, validation_data=validation_generator)

    val_loss, val_accuracy = model.evaluate(validation_generator)

    accuracies.append(val_accuracy)

best_model_index = accuracies.index(max(accuracies))
best_model = models[best_model_index]
best_batch_size = batch_sizes[best_model_index]
best_val_accuracy = accuracies[best_model_index]

model_filename = f'model_batchsize_{best_batch_size}_val_accuracy_{best_val_accuracy:.2f}.h5'

best_model.save(model_filename)