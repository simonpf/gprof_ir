"""
gprof_ir.metrics
===============

Extends the metrics available from the satrain package with spatially resolved metrics.
"""
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from satrain.metrics import QuantificationMetric



class SpatialQuantificationMetric(QuantificationMetric):
    """
    Class representing metrics providing spatially resolved results.
    """


class BiasSpatial(QuantificationMetric):
    r"""
    The bias, or mean error, calculated as the mean value of the difference between
    prediction and target values:

    .. math::

      \\text{Bias} = \\mathbf{E}\{y_\\text{pred} - y_\\text{target}\}

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(
            self,
            resolution = (1.0, 1.0),
            relative: bool = True,
    ):
        """
        Args:
            resolution: The resolution in degree to use for the underlying
                spatial grid. The first element is the resolution along meridional
                direction, the second element the resolution along zonal direction.
            relative: If True, the bias is calculated as percent of the mean reference
                 precipitation. Else the bias is calculated as absolute value.
        """
        self.n_lon = int(360 / resolution[0])
        self.n_lat = int(180 / resolution[1])
        self.bins = (
            np.linspace(-90, 90, self.n_lat + 1),
            np.linspace(-180, 180, self.n_lon + 1)
        )
        self.dims = ("latitude", "longitude")
        self.lats = 0.5 * (self.bins[0][1:] + self.bins[0][:-1])
        self.lons = 0.5 * (self.bins[1][1:] + self.bins[1][:-1])

        super().__init__(
            buffers={
                "x_sum": ((self.n_lat, self.n_lon), np.float64),
                "y_sum": ((self.n_lat, self.n_lon), np.float64),
                "counts": ((self.n_lat, self.n_lon), np.float64),
            }
        )
        self.relative = relative

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             lons: The longitude coordinates of the retrievals.
             lats: The latitude coordinates of the retrievals.
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]

        with self.lock:
            self.x_sum += np.histogram2d(lats, lons, weights=pred, bins=self.bins)[0]
            self.y_sum += np.histogram2d(lats, lons,  weights=target, bins=self.bins)[0]
            self.counts += np.histogram2d(lats, lons, bins=self.bins)[0]

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' containing the
            MSE for the assessed results.
        """
        with np.errstate(invalid="ignore"):
            if self.relative:
                bias = 100.0 * (self.x_sum - self.y_sum) / self.y_sum
            else:
                bias = (self.x_sum - self.y_sum) / self.counts

        bias = xr.Dataset({
            "latitude": (("latitude",), self.lats),
            "longitude": (("longitude",), self.lons),
            "bias_spatial": (("latitude", "longitude"), bias)
        })
        bias.bias_spatial.attrs["full_name"] = "Spatial Bias Distribution"
        bias.bias_spatial.attrs["unit"] = r"\%" if self.relative else "mm h^{-1}"
        return bias


class MAESpatial(QuantificationMetric):
    """
    The mean-absolute error calculated as the mean value of the absolute value
    of the difference between prediction and target values:

    .. math::
      \\text{MAE} = \\mathbf{E}\\{|y_\\text{pred} - y_\\text{target}|\\}.

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(
            self,
            resolution = (1.0, 1.0),
    ):
        """
        Args:
            lons: The longitude coordinates of the retrievals.
            lats: The latitude coordinates of the retrievals.
            resolution: The resolution in degree to use for the underlying
                spatial grid. The first element is the resolution along meridional
        """
        self.n_lon = int(360 / resolution[0])
        self.n_lat = int(180 / resolution[1])
        self.bins = (
            np.linspace(-90, 90, self.n_lat + 1),
            np.linspace(-180, 180, self.n_lon + 1),
        )
        self.dims = ("latitude", "longitude")
        self.lats = 0.5 * (self.bins[0][1:] + self.bins[0][:-1])
        self.lons = 0.5 * (self.bins[1][1:] + self.bins[1][:-1])
        super().__init__(
            buffers={
                "tot_abs_error": ((self.n_lat, self.n_lon), np.float64),
                "counts": ((self.n_lat, self.n_lon), np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
            lons: The longitude coordinates of the retrievals.
            lats: The latitude coordinates of the retrievals.
            prediction: A np.ndarray containing the prediction.
            target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]

        with self.lock:
            self.tot_abs_error += np.histogram2d(
                lats,
                lons,
                weights=np.abs(pred - target),
                bins=self.bins
            )[0]
            self.counts += np.histogram2d(
                lats,
                lons,
                bins=self.bins
            )[0]

    def compute(self) -> xr.Dataset:
        """
        Calculate the MAE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mae' containing
            the MAE for all assessed estimates.
        """
        with np.errstate(invalid="ignore"):
            mae = xr.Dataset({
                "longitude": (("longitude",), self.lons),
                "latitude": (("latitude",), self.lats),
                "mae_spatial": (self.dims, self.tot_abs_error / self.counts)
            })
        mae.mae_spatial.attrs["full_name"] = "Spatial MAE Distribution"
        mae.mae_spatial.attrs["unit"] = "mm h^{-1}"
        return mae


class SMAPESpatial(QuantificationMetric):
    r"""
    The symmetric mean absolute percentage error (SMAPE) with threshold :math:`t`.

    .. math::

      \\text{SMAPE}_t = \\mathbf{E}_{t \\leq y_\\text{target}}\\{\\frac{|y_\\text{pred} - y_\\text{target}|}{ 0.5 (|y_\\text{pred}| + |y_\\text{target}|)}\}

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite and for which the absolute value of the
    exceeds the given threshold value.
    """

    def __init__(
            self,
            resolution: Tuple[float, float] = (1.0, 1.0),
            threshold: float = 0.1,
    ):
        """
        Args:
            lons: The longitude coordinates of the retrievals.
            lats: The latitude coordinates of the retrievals.
            resolution: The resolution in degree to use for the underlying
                spatial grid. The first element is the resolution along meridional
            threshold: Minimum target value for samples to be considered in the
                calculation.

        """
        self.n_lon = int(360 / resolution[0])
        self.n_lat = int(180 / resolution[1])
        self.bins = (
            np.linspace(-90, 90, self.n_lat + 1),
            np.linspace(-180, 180, self.n_lon + 1),
        )
        self.dims = ("latitude", "longitude")
        self.lats = 0.5 * (self.bins[0][1:] + self.bins[0][:-1])
        self.lons = 0.5 * (self.bins[1][1:] + self.bins[1][:-1])
        self.threshold = threshold
        super().__init__(
            buffers={
                "tot_rel_error": ((self.n_lat, self.n_lon), np.float64),
                "counts": ((self.n_lat, self.n_lon), np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: A np.ndarray containing the prediction.
             target: A np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target) * np.abs(target) > self.threshold
        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]

        with self.lock:
            with np.errstate(invalid='ignore'):
                err = np.abs(pred - target) / (0.5 * (np.abs(pred) + np.abs(target)))
                self.tot_rel_error += np.histogram2d(lats, lons, weights=err, bins=self.bins)[0]
                self.counts += np.histogram2d(lats, lons, bins=self.bins)[0]


    def compute(self) -> xr.Dataset:
        """
        Calculate the SMAPE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'smape' representing
            the SMAPE calculated over all results passed to this metric object.

        """
        with np.errstate(invalid='ignore'):
            smape = xr.Dataset({
                "latituide": (("latitude",), self.lats),
                "longitude": (("longitude",), self.lons),
                "smape_spatial": (self.dims, 100.0 * (self.tot_rel_error / self.counts))
            })
        smape.smape_spatial.attrs["full_name"] = f"Spatial SMAPE$_{{{self.threshold:.2}}}$ Distribution"
        smape.smape_spatial.attrs["unit"] = r"\%"
        return smape


class MSESpatial(QuantificationMetric):
    r"""
    The mean-squared error calculated as the mean value of the squared difference between
    prediction and target values:

    .. math::

      \\text{MSE} = (\\mathbf{E}\{y_\\text{pred} - y_\\text{target}\})^2

    where mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(
            self,
            resolution: Tuple[float, float] = (1.0, 1.0)
    ):
        """
        Args:
            lons: The longitude coordinates of the retrievals.
            lats: The latitude coordinates of the retrievals.
            resolution: The resolution in degree to use for the underlying
                spatial grid. The first element is the resolution along meridional
        """
        self.n_lon = int(360 / resolution[0])
        self.n_lat = int(180 / resolution[1])
        self.bins = (
            np.linspace(-90, 90, self.n_lat + 1),
            np.linspace(-180, 180, self.n_lon + 1),
        )
        self.dims = ("latitude", "longitude")
        self.lats = 0.5 * (self.bins[0][1:] + self.bins[0][:-1])
        self.lons = 0.5 * (self.bins[1][1:] + self.bins[1][:-1])
        super().__init__(
            buffers={
                "tot_sq_error": ((self.n_lat, self.n_lon), np.float64),
                "counts": ((self.n_lat, self.n_lon), np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]

        with self.lock:
            self.tot_sq_error += np.histogram2d(
                lats,
                lons,
                weights=(pred - target) ** 2,
                bins=self.bins
            )[0]
            self.counts += np.histogram2d(
                lats,
                lons,
                bins=self.bins
            )[0]

    def compute(self) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' representing
            the MSE calculated over all results passed to this metric object.
        """
        with np.errstate(invalid='ignore'):
            mse = xr.Dataset({
                "latitude": (("latitude",), self.lats),
                "longitude": (("longitude",), self.lons),
                "mse_spatial": (self.dims, self.tot_sq_error / self.counts)
            })
        mse.mse_spatial.attrs["full_name"] = "Spatial MSE Distribution"
        mse.mse_spatial.attrs["unit"] = "(mm h^{-1})^2"
        return mse


class CorrelationCoefSpatial(QuantificationMetric):
    r"""
    The linear correlation coefficient between predictions and target values.

    .. math::

      \\text{Correlation coeff.} = \\mathbf{E}\\frac{
      (y_\\text{pred} - \\mu_{y_\\text{pred}})(y_\\text{target} - \\mu{y_\\text{target})}
      }{
       \\sigma_{y_\\text{pred}} \sigma_{y_\\text{target}}
      }


    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite and :math:`\\mu` and :math:`\\sigma` are used to denote
    the mean and standard deviations of the distributions of :math:`y_\text{pred}` and
    :math:`y_\\text{target}`.
    """
    def __init__(
            self,
            resolution: Tuple[float, float] = (1.0, 1.0)
    ):
        """
        Args:
            resolution: The resolution in degree to use for the underlying
                spatial grid. The first element is the resolution along meridional
        """
        self.n_lon = int(360 / resolution[0])
        self.n_lat = int(180 / resolution[1])
        self.bins = (
            np.linspace(-90, 90, self.n_lat + 1),
            np.linspace(-180, 180, self.n_lon + 1),
        )
        self.dims = ("latitude", "longitude")
        self.lats = 0.5 * (self.bins[0][1:] + self.bins[0][:-1])
        self.lons = 0.5 * (self.bins[1][1:] + self.bins[1][:-1])
        shape = (self.n_lat, self.n_lon)
        super().__init__(
            buffers={
                "x_sum": (shape, np.float64),
                "x2_sum": (shape, np.float64),
                "y_sum": (shape, np.float64),
                "y2_sum": (shape, np.float64),
                "xy_sum": (shape, np.float64),
                "counts": (shape, np.float64),
            }
        )

    def update(
            self,
            lons: np.ndarray,
            lats: np.ndarray,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
            lons: The longitude coordinates of the retrievals.
            lats: The latitude coordinates of the retrievals.
            prediction: An np.ndarray containing the predicted values.
            target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        lons = lons[valid]
        lats = lats[valid]

        with self.lock:
            self.x_sum += np.histogram2d(lats, lons, weights=pred, bins=self.bins)[0]
            self.x2_sum += np.histogram2d(lats, lons, weights=pred ** 2, bins=self.bins)[0]
            self.y_sum += np.histogram2d(lats, lons, weights=target, bins=self.bins)[0]
            self.y2_sum += np.histogram2d(lats, lons, weights=target ** 2, bins=self.bins)[0]
            self.xy_sum += np.histogram2d(lats, lons, weights=pred * target, bins=self.bins)[0]
            self.counts += np.histogram2d(lats, lons, bins=self.bins)[0]

    def compute(self) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        with np.errstate(invalid='ignore'):
            x_mean = self.x_sum / self.counts
            x2_mean = self.x2_sum / self.counts
            x_sigma = np.sqrt(x2_mean - x_mean**2)
            y_mean = self.y_sum / self.counts
            y2_mean = self.y2_sum / self.counts
            y_sigma = np.sqrt(y2_mean - y_mean**2)
            xy_mean = self.xy_sum / self.counts

            # Handle edge case where both variables have zero variance (perfect correlation)
            denominator = x_sigma * y_sigma
            numerator = xy_mean - x_mean * y_mean
            corr = numerator / denominator

            mask = np.isclose(denominator, 0.0, atol=1e-15)
            corr = np.where(mask, np.nan, corr)
            perfect = mask * np.isclose(x_mean, y_mean, atol=1e-15)
            corr = np.where(perfect, 1.0, corr)

        corr = xr.Dataset({
            "latitude": (("latitude",), self.lats),
            "longitude": (("longitue"), self.lons),
            "correlation_coef_spatial": (self.dims, corr)
        })
        corr.correlation_coef_spatial.attrs["full_name"] = "Spatial Correlation Coeff. Distribution"
        corr.correlation_coef_spatial.attrs["unit"] = ""
        return corr



class Calibration(QuantificationMetric):
    r"""
    Calibration plot of predicted quantiles.
    """

    def __init__(
            self,
            tau: np.ndarray
    ):
        """
        Args:
            tau: An array containing the predicted quantile fractions.
        """
        n_quants = tau.size
        self.tau = tau
        super().__init__(
            buffers={
                "less_than": ((n_quants,), np.float64),
                "counts": ((n_quants,), np.float64),
            }
        )

    def update(
            self,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             lons: The longitude coordinates of the retrievals.
             lats: The latitude coordinates of the retrievals.
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        if target.ndim < pred.ndim:
            target = np.expand_dims(target, 1)
        target = np.broadcast_to(target, pred.shape)
        valid = np.isfinite(target)
        target = np.where(valid, target, pred + 1)
        dims = tuple([dim for dim in np.arange(target.ndim) if dim != 1])

        with self.lock:

            self.less_than += (target <= prediction).sum(axis=dims)
            self.counts += valid.sum(dims).astype(np.float32)


    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' containing the
            MSE for the assessed results.
        """
        with np.errstate(invalid="ignore"):
            calibration = self.less_than / self.counts

        calibration = xr.Dataset({
            "tau": (("tau",), self.tau),
            "calibration": (("calibration",), calibration),
        })
        calibration.calibration["full_name"] = "Calibration"
        calibration.calibration.attrs["unit"] = r"Fraction"
        return calibration


class DetectionCalibration(QuantificationMetric):
    r"""
    Calibration plot of predicted quantiles.
    """

    def __init__(
            self,
    ):
        """
        Args:
            tau: An array containing the predicted quantile fractions.
        """
        self.bins = np.linspace(0.1, 1, 21)
        self.levels = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.n_levels = self.levels.size
        super().__init__(
            buffers={
                "right": ((self.n_levels,), np.float64),
                "counts": ((self.n_levels,), np.float64),
            }
        )

    def update(
            self,
            prediction: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Update metric values with given prediction.

        Args:
             lons: The longitude coordinates of the retrievals.
             lats: The latitude coordinates of the retrievals.
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        valid = np.isfinite(target)
        pred = prediction[valid]
        target = target[valid]

        with self.lock:
            self.right += np.histogram(pred, weights=target, bins=self.bins)[0]
            self.counts += np.histogram(pred, bins=self.bins)[0]


    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' containing the
            MSE for the assessed results.
        """
        with np.errstate(invalid="ignore"):
            calibration = self.right / self.counts

        calibration = xr.Dataset({
            "levels": (("levels",), self.levels),
            "calibration": (("levels",), calibration),
        })
        calibration.calibration["full_name"] = "Detetion Calibration"
        calibration.calibration.attrs["unit"] = r"Fraction"
        return calibration
