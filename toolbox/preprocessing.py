import numpy as np
import SimpleITK as sitk


def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    """ Resample image to  isotropic resolution.
    :param itk_image - SimpleITK image.
    :param out_spacing - desired resampling resolution
    :return - resampled SimpleITK image
    """
    original_spacing = itk_image.GetSpacing()  # Takes into account Z-axis (i.e. slice thickness) as well
    original_size = itk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())  # Padding

    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)


def resample_image_to_ref(image, ref, size=(256,256,256), bspline=False, pad_value=0):
    """ Resamples an image to match the resolution and size of a given reference image.
    :param image: SimpleITK image object - target image.
    :param ref:  SimpleITK image object - reference image.
    :param size: desired output size.
    :param bspline: use BSpline interpolation or not.
    :param pad_value: value of the padded pixels
    :return: resampled image.
    """

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)
    resample.SetSize(size)

    if bspline:
        resample.SetInterpolator(sitk.sitkBSpline)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(image)
