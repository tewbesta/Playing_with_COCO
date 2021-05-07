import os
import json
import requests
from PIL import Image, UnidentifiedImageError
with open('imagenet_class_info.json') as f:
  jsondata = json.load(f)

import argparse
parser = argparse.ArgumentParser( description ='HW02 Task1')
parser.add_argument('--subclass_list', nargs ='*',type =str, required = True)
parser.add_argument('--images_per_subclass', type =int ,required = True )
parser.add_argument('--data_root', type=str, required =True )
parser.add_argument('--main_class',type =str, required =True )
parser.add_argument('--imagenet_info_json', type =str,required =True )
args, args_other = parser.parse_known_args()
print(args.images_per_subclass)
print(args.subclass_list)

        #from PIL import Image
from requests.exceptions import ConnectionError,ReadTimeout,TooManyRedirects,MissingSchema,InvalidURL
count=0
def get_image(img_url, class_folder):
           global count
           if len(img_url) <= 1:
              print("length is samll")
            #url is useless Do something
           try:
             img_resp = requests.get(img_url, timeout=1)
           except ConnectionError:
              print("ConnectionError")
              return "fail"
            # Handle this exception
              img_resp="No response"
           except ReadTimeout:
             print("ReadTimeout")
             return "fail"
            # Handle this exception
           except TooManyRedirects:
             print("TooManyRedirects")
             return "fail"
            # handle exception
           except MissingSchema:
             print("MissingSchema")
             return "fail"
            # handle exception
           except InvalidURL:
            # handle exception
             print("InvalidURL")
             return "fail"

           if not 'content-type' in img_resp.headers:
                    # Missing content . Do something
                    print("imageheader missing cont")
                    return "fail"
           if not 'image' in img_resp.headers['content-type']:
              # The url doesn ’t have any image . Do something
                  print("imageheader")
                  return "fail"
           if (len(img_resp.content) < 1000):
               # ignore images < 1kb
                    print("length")
                    return "fail"
           img_name = img_url.split('/')[-1]
           img_name = img_name.split("?")[0]
           if (len(img_name) <= 1):
            # missing image name
              print("name")
              return "fail"
           if not 'flickr' in img_url:
              print("flickr")
              return "fail"
            # Missing non - flickr images are difficult to handle.Dosomething.img_file_path = os.path.join(class_folder, img_name)
           img_file_path = os.path.join(class_folder, img_name)
           with open(img_file_path, 'wb') as img_f:
             img_f.write(img_resp.content)
            # Resize image to 64x64
           try:
               im = Image.open(img_file_path)
           except UnidentifiedImageError:
               print ("Unidentified image")
               return "fail"
           if im.mode != "RGB ":
                 im = im.convert(mode="RGB")
           im_resized = im.resize((64, 64), Image.BOX)
          # Overwrite original image with downsampled image
           im_resized.save(img_file_path)

           count=count+1

for key, value in jsondata.items():
        the_list_url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
        for i in range(len(args.subclass_list)):

             count=0
             if (value["class_name"] == args.subclass_list[i]):
                        print(value["class_name"])
                        print(the_list_url + key)
                        the_list_url = the_list_url + key
                        resp = requests.get(the_list_url)
                        urls = [url.decode('utf -8') for url in resp.content.splitlines()]
                        Train_val=(args.data_root)
                        print(Train_val)
                        if not os.path.exists(args.data_root):
                            os.mkdir(args.data_root)
                        for url in urls:
                         if count<=args.images_per_subclass:
                            print(count)
                            if args.main_class=="cat":
                              if Train_val == "Train/":
                                if not os.path.exists(os.path.join(args.data_root,"cat")):
                                    path=os.path.join(args.data_root,"cat")
                                    os.mkdir(path,mode = 0o666)
                                get_image(url, args.data_root+"cat")

                              if Train_val=="Val/":
                                  if not os.path.exists(args.data_root+ "cat"):
                                      os.mkdir(args.data_root + "cat")
                                  get_image(url, args.data_root+"cat")
                            if args.main_class == "dog":

                              if Train_val == "Train/":
                                if not os.path.exists(os.path.join(args.data_root, "dog")):
                                        os.mkdir(args.data_root+"dog")
                                get_image(url, args.data_root + "dog")

                              if Train_val=="Val/":
                                 if not os.path.exists(os.path.join(args.data_root, "dog")):
                                        os.mkdir(args.data_root + "dog")
                                 get_image(url, args.data_root+"dog")



