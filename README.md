# Playing_with_COCO Downloading and Classifying images from COCO_Dataset
Download the images and start classifying cats and dogs images. After running the COCO_Downloader, you should be able to see images downloaded in its own folder.

python COCO_image_downloader.py --subclass_list ‘Siamese cat’ ‘Persian cat’ ‘Burmese cat’ \
--main_class ‘cat’ --data_root <imagenet_root>/Train/ \
--imagenet_info_json <path_to_imagenet_class_info.json> --images_per_subclass 200

python COCO_image_downloader.py --subclass_list ‘hunting dog’ ‘sporting dog’ ‘shepherd dog’ \
--main_class ‘dog’ --data_root <imagenet_root>/Train/ \
--imagenet_info_json <path_to_imagenet_class_info.json> --images_per_subclass 200

python COCO_image_downloader.py --subclass_list ‘domestic cat’ ‘alley cat’ \
--main_class ‘cat’ --data_root <imagenet_root>/Val/ \
--imagenet_info_json <path_to_imagenet_class_info.json> --images_per_subclass 100

python COCO_image_downloader.py --subclass_list ‘working dog’ ‘police dog’ \
--main_class ‘dog’ --data_root <imagenet_root>/Val/ \
--imagenet_info_json <path_to_imagenet_class_info.json> --images_per_subclass 100

Feel free to change the --images_per_subclass in the COCO_Downloader to the amount of images you want to download.

python COCO_classification.py --imagenet_root <path_to_imagenet_root> --class_list 'cat' 'dog'

Make sure you change <path_to_imagenet_root> to the path where your images from COCO_downloader are saved.

