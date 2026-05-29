import rdv
import grid_implementations as imp
import torch
from vulky import datasets
import matplotlib.pyplot as plt


vol = datasets.Volumes.disney_cloud


plt.imshow(vol[:,:, vol.shape[2]//2, 0].T.cpu())
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.show()


# this is for performance, copying the tensor in rdv memory.
vol = rdv.tensor_copy(vol)

# creates the grid map
grid = imp.ExperimentalGrid3D(vol)
# gets a fitting box for the grid, assuming the max dimension will fit -1,1 and
# aspect ratio is preserved.
bmin, bmax = rdv.grid3d_fit_box(vol.shape[:3])
# scale and offset required for converting normalized coordinates into the fit box
scale, offset = rdv.unit2box(bmin, bmax)
# create the transform
transform = rdv.mat4x3.trs(offset=offset, scale=scale)

# Creates a transmittance map that receives 6 values: ray position (3), ray direction (3)
# and returns the transmittance (0..1). 1 indicates no particles where found so all photons travel without interruption.
transmittance_map = rdv.RaymarchingTransmittance(
    extinction=grid * 5,  # notice this multiplication is not to the tensor but to the map.
    step_size=0.001,
    transform=transform
)

# create camera_pose
camera_poses = rdv.tensor_from(
    [
        # camera pose 0
        [
            0.2, 0.3, -2.5, # position
            0.0, 0.0, 0.0,  # target
            0.0, 1.0, 0.0,  # up vector
        ]
    ]
)

# Create a sensor object to represent the camera
sensor = rdv.Sensor(1, 512, 512,
                    # first index is the camera index and is sampled in the corner/floor (each sample correspond exactly to the index)
                    # second and third indices are treated as pixels py and px, position is sampled in the center of the pixel
                    samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.CENTER, rdv.SampleLocation.CENTER),
                    # probes map are the map in charge of converting a sensor index into a sample, in this case
                    # a ray sample with position and direction is generated for each pixel of the camera using the camera pose.
                    probes_map=rdv.CameraProbes(camera_poses=camera_poses)
)

# Creates a map that is a view/sample of the field (transmittance_map) as seen/sampled from the sensor
transmittance_view = sensor.view(transmittance_map)

with torch.no_grad():
    # captures a sample, an image with shape (1, 512, 512) imposed by the sensor.
    transmittance_img = transmittance_view.capture()

#view the transmittance image
plt.imshow(transmittance_img[0].cpu(), vmin=0.0, vmax=1.0, cmap='Blues_r')
plt.gca().axis('off')
plt.gca().invert_yaxis()
plt.tight_layout(pad=0.0)
plt.show()


