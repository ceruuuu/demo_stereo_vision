import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    color_raw = o3d.io.read_image("./images/imgleft.png")
    depth_raw = o3d.io.read_image("./images/filteredImg-16.png")

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0, depth_trunc=1000000, convert_rgb_to_intensity=False)

    plt.subplot(1, 2, 1)
    plt.title('grayscale image')
    plt.imshow(rgbd_image.color)

    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd_image.depth)

    plt.show()

    # color, depth -> <class 'open3d.cpu.pybind.geometry.Image'>
    as_color = np.asarray(rgbd_image.color, dtype=float, order=None)
    as_depth = np.asarray(rgbd_image.depth, dtype=float, order=None)

    h, w = as_depth.shape

    # point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(w, h, w/2, h/2, w/2, h/2))
            #o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Flip it, otherwise the pointcloud will be upside down
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])