from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator

original_image, input_image = prepare_image(image_pth='examples/2616_contact.png')  # change the image path to your image

mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type='L', ckpt='model/swinl_only_sam_many2many.pth')) # model_type: 'L' / 'T', depends on your checkpint
masks = mask_generator.generate(input_image)

plot_results(masks, original_image, save_path='./vis/')  # results and original images will be saved at save_path