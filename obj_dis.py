import sys
import numpy as np
import pyzed.sl as sl
import cv2
import math
help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)
    

def process_key_event(zed, key) :
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")


def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    #init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking=True
    obj_param.enable_segmentation=True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
    
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))

    if obj_param.enable_tracking :
        positional_tracking_param = sl.PositionalTrackingParameters()
        #positional_tracking_param.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Object Detection: Loading Module...")

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS :
        print("Enable object detection : "+repr(err)+". Exit program.")
        zed.close()
        exit()

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40
    
    
    
    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    #depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.F32_C1)

    point_cloud = sl.Mat()

    key = ' '
    while key != 113 :
        err = zed.grab(runtime)
        # Create a sl.Mat with float type (32-bit)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            #zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            #depth_image_ocv = depth_image_zed.get_data()
            # Retrieve depth data (32-bit)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            # Load depth data into a numpy array
            depth_ocv = depth_zed.get_data()
            # Print the depth value at the center of the image
            #print(depth_ocv[int(len(depth_ocv)/2)][int(len(depth_ocv[0])/2)])


            
            #cv2.imshow("Depth", depth_image_ocv)
            zed.retrieve_objects(objects, obj_runtime_param)
            if objects.is_new :
                obj_array = objects.object_list
                print(str(len(obj_array))+" Object(s) detected\n")
                if len(obj_array) > 0 :
                    for item in obj_array:
                        first_object = item
                        print("First object attributes:")
                        print(" Label '"+repr(first_object.label)+"' (conf. "+str(int(first_object.confidence))+"/100)")
                        if obj_param.enable_tracking :
                            print(" Tracking ID: "+str(int(first_object.id))+" tracking state: "+repr(first_object.tracking_state)+" / "+repr(first_object.action_state))
                        position = first_object.position
                        velocity = first_object.velocity
                        dimensions = first_object.dimensions
                        #print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(position[0],position[1],position[2],velocity[0],velocity[1],velocity[2],dimensions[0],dimensions[1],dimensions[2]))
                        if first_object.mask.is_init():
                            print(" 2D mask available")

                        #print(" Bounding Box 2D ")
                        bounding_box_2d = first_object.bounding_box_2d
                        x,y=(bounding_box_2d[0].astype(int)+bounding_box_2d[2].astype(int))/2
                        err, point_cloud_value = point_cloud.get_value(int(x), int(y))
                        if math.isfinite(point_cloud_value[2]):
                            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                                point_cloud_value[1] * point_cloud_value[1] +
                                                point_cloud_value[2] * point_cloud_value[2])
                            s=str(round(distance/1000,2))+'m'
                            cv2.rectangle(image_ocv,bounding_box_2d[0].astype(int),bounding_box_2d[2].astype(int),(0,0,255),2,cv2.LINE_AA)
                            cv2.putText(image_ocv,repr(first_object.label),bounding_box_2d[0].astype(int)-30,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                            cv2.putText(image_ocv,s,bounding_box_2d[0].astype(int)+30,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                            for it in bounding_box_2d :
                                print("    "+str(it),end='')
                            #print("\n Bounding Box 3D ")
                            bounding_box = first_object.bounding_box
                            for it in bounding_box :
                                print("    "+str(it),end='')
                            cv2.imshow("Image", image_ocv)
            key = cv2.waitKey(10)

            process_key_event(zed, key)
            

    cv2.destroyAllWindows()
    zed.disable_object_detection()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()