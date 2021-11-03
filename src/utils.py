import matplotlib.pyplot as plt

# A method to generate a sample image on which we can see the training efficiency visually
def generate_and_save_images(model, test_images, epoch):

    predictions = model(test_images, training=False)

    fig = plt.figure(figsize=(20, 20))
    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255.0, cmap='gray')
        plt.axis('off')
    plt.savefig('../Logs/output_images/image_at_epoch_{:04d}.png'.format(epoch))