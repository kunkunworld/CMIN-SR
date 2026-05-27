from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DRegressor(nn.Module):
    def __init__(self, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.act(x)


class ResNet1DRegressor(nn.Module):
    def __init__(self, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32, stride=1, dropout=dropout),
            ResidualBlock1D(32, 64, stride=2, dropout=dropout),
            ResidualBlock1D(64, 64, stride=1, dropout=dropout),
            ResidualBlock1D(64, 128, stride=2, dropout=dropout),
            ResidualBlock1D(128, 128, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


class ScaleInvariantCNNRegressor(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        moment_qs: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.register_buffer("moment_qs", torch.tensor(moment_qs, dtype=torch.float32))

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, feature_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.scale_embedding = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.scale_attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )
        self.moment_projection = nn.Sequential(
            nn.LayerNorm(len(scales) * len(moment_qs)),
            nn.Linear(len(scales) * len(moment_qs), feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
        )

    def _scaled_view(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        x = x.unsqueeze(1)
        if scale == 1:
            return x
        return F.avg_pool1d(x, kernel_size=scale, stride=scale, ceil_mode=False)

    def _moment_features(self, x: torch.Tensor) -> torch.Tensor:
        moments = []
        qs = self.moment_qs.view(1, -1, 1)
        for scale in self.scales:
            x_scale = self._scaled_view(x, scale) * float(scale)
            abs_x = x_scale.abs().clamp_min(1e-6)
            moment = torch.log(torch.mean(abs_x.pow(qs), dim=-1) + 1e-6)
            moments.append(moment)
        return torch.cat(moments, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_features = []
        for scale in self.scales:
            x_scale = self._scaled_view(x, scale)
            feature = self.encoder(x_scale)
            log_scale = x.new_full((x.shape[0], 1), float(scale)).log()
            feature = feature + self.scale_embedding(log_scale)
            scale_features.append(feature)

        features = torch.stack(scale_features, dim=1)
        weights = torch.softmax(self.scale_attention(features), dim=1)
        pooled = torch.sum(features * weights, dim=1)
        moment_feature = self.moment_projection(self._moment_features(x))
        return self.head(torch.cat([pooled, moment_feature], dim=1))


class PhysicsScaleNetRegressor(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        feature_dim: int = 160,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.register_buffer("moment_qs", torch.tensor(moment_qs, dtype=torch.float32))

        log_scales = torch.log(torch.tensor(scales, dtype=torch.float32))
        centered_log_scales = log_scales - log_scales.mean()
        self.register_buffer("centered_log_scales", centered_log_scales)
        self.register_buffer("scale_slope_denom", torch.sum(centered_log_scales.square()).clamp_min(1e-6))

        self.raw_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, feature_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.raw_scale_embedding = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.raw_scale_attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )

        self.surface_encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        prior_dim = len(moment_qs) * 3
        self.prior_encoder = nn.Sequential(
            nn.LayerNorm(prior_dim),
            nn.Linear(prior_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2 + 48),
            nn.Linear(feature_dim * 2 + 48, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, output_dim),
        )

    def _aggregate_increments(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        x = x.unsqueeze(1)
        if scale == 1:
            return x
        return F.avg_pool1d(x, kernel_size=scale, stride=scale, ceil_mode=False) * float(scale)

    def _raw_scale_features(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for scale in self.scales:
            x_scale = self._aggregate_increments(x, scale)
            feature = self.raw_encoder(x_scale)
            log_scale = x.new_full((x.shape[0], 1), float(scale)).log()
            features.append(feature + self.raw_scale_embedding(log_scale))
        features_t = torch.stack(features, dim=1)
        weights = torch.softmax(self.raw_scale_attention(features_t), dim=1)
        return torch.sum(features_t * weights, dim=1)

    def _log_moment_surface(self, x: torch.Tensor) -> torch.Tensor:
        qs = self.moment_qs.view(1, 1, -1)
        moments = []
        for scale in self.scales:
            aggregated = self._aggregate_increments(x, scale).abs().transpose(1, 2).clamp_min(1e-6)
            moments.append(torch.log(torch.mean(aggregated.pow(qs), dim=1) + 1e-6))
        return torch.stack(moments, dim=1)

    def _physics_prior_features(self, log_moments: torch.Tensor) -> torch.Tensor:
        centered_surface = log_moments - log_moments.mean(dim=1, keepdim=True)
        slopes = torch.sum(
            centered_surface * self.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.scale_slope_denom
        q = self.moment_qs.view(1, -1)
        h_proxy = slopes / q.clamp_min(1e-6)
        curvature = torch.zeros_like(slopes)
        curvature[:, 1:-1] = slopes[:, 2:] - 2.0 * slopes[:, 1:-1] + slopes[:, :-2]
        return torch.cat([slopes, h_proxy, curvature], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_feature = self._raw_scale_features(x)
        log_moments = self._log_moment_surface(x)
        centered_surface = log_moments - log_moments.mean(dim=1, keepdim=True)
        surface_feature = self.surface_encoder(centered_surface.unsqueeze(1))
        prior_feature = self.prior_encoder(self._physics_prior_features(log_moments))
        return self.head(torch.cat([raw_feature, surface_feature, prior_feature], dim=1))


class PhysicsHybridCNNRegressor(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        prior_dim: int = 128,
    ) -> None:
        super().__init__()
        self.base = CNN1DRegressor(output_dim=output_dim, dropout=dropout)
        self.scales = scales
        self.register_buffer("moment_qs", torch.tensor(moment_qs, dtype=torch.float32))

        log_scales = torch.log(torch.tensor(scales, dtype=torch.float32))
        centered_log_scales = log_scales - log_scales.mean()
        self.register_buffer("centered_log_scales", centered_log_scales)
        self.register_buffer("scale_slope_denom", torch.sum(centered_log_scales.square()).clamp_min(1e-6))

        num_q = len(moment_qs)
        prior_features = num_q * 4
        self.prior = nn.Sequential(
            nn.LayerNorm(prior_features),
            nn.Linear(prior_features, prior_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prior_dim, prior_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prior_dim, output_dim),
        )
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)

    def _aggregate_increments(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        x = x.unsqueeze(1)
        if scale == 1:
            return x
        return F.avg_pool1d(x, kernel_size=scale, stride=scale, ceil_mode=False) * float(scale)

    def _log_moment_surface(self, x: torch.Tensor) -> torch.Tensor:
        qs = self.moment_qs.view(1, 1, -1)
        moments = []
        for scale in self.scales:
            aggregated = self._aggregate_increments(x, scale).abs().transpose(1, 2).clamp_min(1e-6)
            moments.append(torch.log(torch.mean(aggregated.pow(qs), dim=1) + 1e-6))
        return torch.stack(moments, dim=1)

    def _prior_features(self, x: torch.Tensor) -> torch.Tensor:
        log_moments = self._log_moment_surface(x)
        centered_surface = log_moments - log_moments.mean(dim=1, keepdim=True)
        slopes = torch.sum(
            centered_surface * self.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.scale_slope_denom
        q = self.moment_qs.view(1, -1)
        h_proxy = slopes / q.clamp_min(1e-6)
        curvature = torch.zeros_like(slopes)
        curvature[:, 1:-1] = slopes[:, 2:] - 2.0 * slopes[:, 1:-1] + slopes[:, :-2]
        roughness = log_moments.std(dim=1)
        return torch.cat([slopes, h_proxy, curvature, roughness], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_pred = self.base(x)
        correction = self.prior(self._prior_features(x))
        return base_pred + correction


class WaveletPhysicsHybridRegressor(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        prior_dim: int = 160,
    ) -> None:
        super().__init__()
        self.base = CNN1DRegressor(output_dim=output_dim, dropout=dropout)
        self.scales = scales
        self.register_buffer("moment_qs", torch.tensor(moment_qs, dtype=torch.float32))

        log_scales = torch.log(torch.tensor(scales, dtype=torch.float32))
        centered_log_scales = log_scales - log_scales.mean()
        self.register_buffer("centered_log_scales", centered_log_scales)
        self.register_buffer("scale_slope_denom", torch.sum(centered_log_scales.square()).clamp_min(1e-6))

        self.surface_encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        num_q = len(moment_qs)
        prior_features = num_q * 5 + 48
        self.prior = nn.Sequential(
            nn.LayerNorm(prior_features),
            nn.Linear(prior_features, prior_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prior_dim, prior_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prior_dim, output_dim),
        )
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)

    def _haar_wavelet_response(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        half = scale
        kernel = x.new_ones(1, 1, half * 2)
        kernel[:, :, :half] = 1.0 / float(half)
        kernel[:, :, half:] = -1.0 / float(half)
        x_1d = x.unsqueeze(1)
        return F.conv1d(x_1d, kernel, stride=half, padding=0).abs().clamp_min(1e-6)

    def _wavelet_log_moment_surface(self, x: torch.Tensor) -> torch.Tensor:
        qs = self.moment_qs.view(1, 1, -1)
        moments = []
        for scale in self.scales:
            response = self._haar_wavelet_response(x, scale).transpose(1, 2)
            moment = torch.log(torch.mean(response.pow(qs), dim=1) + 1e-6)
            moments.append(moment)
        return torch.stack(moments, dim=1)

    def _prior_features(self, x: torch.Tensor) -> torch.Tensor:
        log_moments = self._wavelet_log_moment_surface(x)
        centered_surface = log_moments - log_moments.mean(dim=1, keepdim=True)
        slopes = torch.sum(
            centered_surface * self.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.scale_slope_denom
        q = self.moment_qs.view(1, -1)
        h_proxy = slopes / q.clamp_min(1e-6)
        curvature = torch.zeros_like(slopes)
        curvature[:, 1:-1] = slopes[:, 2:] - 2.0 * slopes[:, 1:-1] + slopes[:, :-2]
        scale_variability = log_moments.std(dim=1)
        q_variability = log_moments.std(dim=2)
        q_variability_summary = F.interpolate(
            q_variability.unsqueeze(1),
            size=slopes.shape[1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        surface_feature = self.surface_encoder(centered_surface.unsqueeze(1))
        return torch.cat(
            [slopes, h_proxy, curvature, scale_variability, q_variability_summary, surface_feature],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_pred = self.base(x)
        correction = self.prior(self._prior_features(x))
        return base_pred + correction


class MRWStatisticalFrontend(nn.Module):
    def __init__(
        self,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        common_length: int = 64,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.common_length = common_length
        self.register_buffer("moment_qs", torch.tensor(moment_qs, dtype=torch.float32))

        log_scales = torch.log(torch.tensor(scales, dtype=torch.float32))
        centered_log_scales = log_scales - log_scales.mean()
        self.register_buffer("log_scales", log_scales)
        self.register_buffer("centered_log_scales", centered_log_scales)
        self.register_buffer("scale_slope_denom", torch.sum(centered_log_scales.square()).clamp_min(1e-6))

    def _aggregated_increments(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        kernel = x.new_ones(1, 1, scale)
        return F.conv1d(x.unsqueeze(1), kernel, stride=scale, padding=0).squeeze(1)

    def _haar_response(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        half = scale
        kernel = x.new_ones(1, 1, half * 2)
        kernel[:, :, :half] = 1.0 / float(half)
        kernel[:, :, half:] = -1.0 / float(half)
        return F.conv1d(x.unsqueeze(1), kernel, stride=half, padding=0).squeeze(1)

    def _moment_surface(self, x: torch.Tensor) -> torch.Tensor:
        qs = self.moment_qs.view(1, 1, -1)
        rows = []
        for scale in self.scales:
            inc = self._aggregated_increments(x, scale).abs().unsqueeze(-1).clamp_min(1e-6)
            rows.append(torch.log(torch.mean(inc.pow(qs), dim=1) + 1e-6))
        return torch.stack(rows, dim=1)

    def _slope_surface(self, moment_surface: torch.Tensor) -> torch.Tensor:
        slopes = torch.zeros_like(moment_surface)
        log_s = self.log_scales
        slopes[:, 0, :] = (moment_surface[:, 1, :] - moment_surface[:, 0, :]) / (log_s[1] - log_s[0])
        slopes[:, -1, :] = (moment_surface[:, -1, :] - moment_surface[:, -2, :]) / (log_s[-1] - log_s[-2])
        slopes[:, 1:-1, :] = (
            moment_surface[:, 2:, :] - moment_surface[:, :-2, :]
        ) / (log_s[2:] - log_s[:-2]).view(1, -1, 1)
        return slopes

    def _zeta_proxy(self, moment_surface: torch.Tensor) -> torch.Tensor:
        centered = moment_surface - moment_surface.mean(dim=1, keepdim=True)
        return torch.sum(centered * self.centered_log_scales.view(1, -1, 1), dim=1) / self.scale_slope_denom

    def _q_curvature(self, zeta_proxy: torch.Tensor) -> torch.Tensor:
        curvature = torch.zeros_like(zeta_proxy)
        curvature[:, 1:-1] = zeta_proxy[:, 2:] - 2.0 * zeta_proxy[:, 1:-1] + zeta_proxy[:, :-2]
        return curvature

    def _wavelet_cumulants_and_corr(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cumulants = []
        aligned_logs = []
        for scale in self.scales:
            log_abs = torch.log(self._haar_response(x, scale).abs().clamp_min(1e-6))
            mean = log_abs.mean(dim=1)
            centered = log_abs - mean.unsqueeze(1)
            var = centered.square().mean(dim=1)
            third = centered.pow(3).mean(dim=1)
            cumulants.append(torch.stack([mean, var, third], dim=1))
            aligned_logs.append(F.adaptive_avg_pool1d(log_abs.unsqueeze(1), self.common_length).squeeze(1))

        cumulants_t = torch.stack(cumulants, dim=1)
        logs = torch.stack(aligned_logs, dim=1)
        logs = logs - logs.mean(dim=2, keepdim=True)
        denom = torch.sqrt(torch.sum(logs.square(), dim=2, keepdim=True).clamp_min(1e-6))
        logs_norm = logs / denom
        corr = torch.bmm(logs_norm, logs_norm.transpose(1, 2))
        return cumulants_t, corr

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        moment_surface = self._moment_surface(x)
        slope_surface = self._slope_surface(moment_surface)
        zeta_proxy = self._zeta_proxy(moment_surface)
        q_curvature = self._q_curvature(zeta_proxy)
        wavelet_cumulants, cross_scale_corr = self._wavelet_cumulants_and_corr(x)
        return {
            "moment_surface": moment_surface,
            "slope_surface": slope_surface,
            "zeta_proxy": zeta_proxy,
            "q_curvature": q_curvature,
            "wavelet_cumulants": wavelet_cumulants,
            "cross_scale_corr": cross_scale_corr,
        }


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            diff = y.shape[-1] - x.shape[-1]
            left = diff // 2
            y = y[:, :, left:left + x.shape[-1]]
        return self.act(x + y)


class PCSMINRegressor(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        hidden_dim: int = 160,
        use_wavelet_cumulants: bool = True,
        use_cross_scale_corr: bool = True,
        use_raw_tcn: bool = True,
    ) -> None:
        super().__init__()
        if output_dim != 2:
            raise ValueError("PCSMINRegressor currently supports parameter mode with output_dim=2.")

        self.frontend = MRWStatisticalFrontend(scales=scales, moment_qs=moment_qs)
        self.use_wavelet_cumulants = use_wavelet_cumulants
        self.use_cross_scale_corr = use_cross_scale_corr
        self.use_raw_tcn = use_raw_tcn
        num_scales = len(scales)
        num_q = len(moment_qs)
        low_q_count = sum(q <= 2.0 for q in moment_qs)

        roughness_dim = low_q_count + 2 * num_scales + 6
        self.roughness_branch = nn.Sequential(
            nn.LayerNorm(roughness_dim),
            nn.Linear(roughness_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.intermittency_surface = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.intermittency_branch = nn.Sequential(
            nn.LayerNorm(64 + num_q * 2),
            nn.Linear(64 + num_q * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.cross_scale_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )

        raw_channels = 48
        self.raw_stem = nn.Sequential(
            nn.Conv1d(1, raw_channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(raw_channels),
            nn.GELU(),
        )
        self.raw_tcn = nn.Sequential(
            TCNBlock(raw_channels, dilation=1, dropout=dropout),
            TCNBlock(raw_channels, dilation=2, dropout=dropout),
            TCNBlock(raw_channels, dilation=4, dropout=dropout),
            TCNBlock(raw_channels, dilation=8, dropout=dropout),
            TCNBlock(raw_channels, dilation=16, dropout=dropout),
        )
        self.raw_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(raw_channels, hidden_dim),
            nn.GELU(),
        )

        self.h_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lambda_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _roughness_features(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        zeta_proxy = stats["zeta_proxy"]
        moment_surface = stats["moment_surface"]
        wavelet_cumulants = stats["wavelet_cumulants"]
        if not self.use_wavelet_cumulants:
            wavelet_cumulants = torch.zeros_like(wavelet_cumulants)
        low_q = zeta_proxy[:, :7]
        q1_slope = stats["slope_surface"][:, :, 2]
        q2_slope = stats["slope_surface"][:, :, 6]
        cumulant_mean = wavelet_cumulants.mean(dim=1)
        cumulant_slope = torch.sum(
            (wavelet_cumulants - wavelet_cumulants.mean(dim=1, keepdim=True))
            * self.frontend.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.frontend.scale_slope_denom
        del moment_surface
        return torch.cat([low_q, q1_slope, q2_slope, cumulant_mean, cumulant_slope], dim=1)

    def _intermittency_features(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        surface = torch.stack([stats["moment_surface"], stats["slope_surface"]], dim=1)
        surface_feature = self.intermittency_surface(surface)
        return torch.cat([surface_feature, stats["zeta_proxy"], stats["q_curvature"]], dim=1)

    def _raw_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.raw_stem(x.unsqueeze(1))
        y = self.raw_tcn(y)
        return self.raw_projection(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.frontend(x)
        roughness_repr = self.roughness_branch(self._roughness_features(stats))
        intermittency_repr = self.intermittency_branch(self._intermittency_features(stats))
        cross_corr = stats["cross_scale_corr"]
        if not self.use_cross_scale_corr:
            cross_corr = torch.zeros_like(cross_corr)
        cross_repr = self.cross_scale_branch(cross_corr.unsqueeze(1))
        raw_repr = self._raw_features(x)
        if not self.use_raw_tcn:
            raw_repr = torch.zeros_like(raw_repr)

        h_raw = self.h_head(torch.cat([roughness_repr, raw_repr], dim=1))
        lambda_raw = self.lambda_head(torch.cat([intermittency_repr, cross_repr, raw_repr], dim=1))

        return torch.cat([lambda_raw, h_raw], dim=1)


class CrossScaleGraphAttention(nn.Module):
    def __init__(
        self,
        num_scales: int,
        hidden_dim: int = 160,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_dim = num_scales + 3 + 1
        self.node_projection = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        corr: torch.Tensor,
        wavelet_cumulants: torch.Tensor,
        log_scales: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = corr.shape[0]
        scale_feature = log_scales.view(1, -1, 1).expand(batch_size, -1, -1)
        nodes = torch.cat([corr, wavelet_cumulants, scale_feature], dim=2)
        nodes = self.node_projection(nodes)

        attended, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        nodes = self.attention_norm(nodes + attended)
        nodes = self.ffn_norm(nodes + self.ffn(nodes))

        weights = torch.softmax(self.pool_score(nodes), dim=1)
        pooled_mean = torch.sum(nodes * weights, dim=1)
        pooled_std = torch.sqrt(torch.sum(weights * (nodes - pooled_mean.unsqueeze(1)).square(), dim=1).clamp_min(1e-6))
        return self.output(torch.cat([pooled_mean, pooled_std], dim=1))


class PCSMINV2Regressor(PCSMINRegressor):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        hidden_dim: int = 160,
        use_raw_gates: bool = True,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            scales=scales,
            moment_qs=moment_qs,
            hidden_dim=hidden_dim,
        )
        self.cross_scale_branch = CrossScaleGraphAttention(
            num_scales=len(scales),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.use_raw_gates = use_raw_gates
        self.h_raw_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.lambda_raw_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.frontend(x)
        roughness_repr = self.roughness_branch(self._roughness_features(stats))
        intermittency_repr = self.intermittency_branch(self._intermittency_features(stats))
        cross_corr = stats["cross_scale_corr"]
        if not self.use_cross_scale_corr:
            cross_corr = torch.zeros_like(cross_corr)
        wavelet_cumulants = stats["wavelet_cumulants"]
        if not self.use_wavelet_cumulants:
            wavelet_cumulants = torch.zeros_like(wavelet_cumulants)
        cross_repr = self.cross_scale_branch(cross_corr, wavelet_cumulants, self.frontend.log_scales)

        raw_repr = self._raw_features(x)
        if not self.use_raw_tcn:
            raw_repr = torch.zeros_like(raw_repr)

        if self.use_raw_gates:
            h_gate = self.h_raw_gate(torch.cat([roughness_repr, raw_repr], dim=1))
            lambda_gate = self.lambda_raw_gate(torch.cat([intermittency_repr, cross_repr, raw_repr], dim=1))
            h_raw_repr = raw_repr * h_gate
            lambda_raw_repr = raw_repr * lambda_gate
        else:
            h_raw_repr = raw_repr
            lambda_raw_repr = raw_repr

        h_raw = self.h_head(torch.cat([roughness_repr, h_raw_repr], dim=1))
        lambda_raw = self.lambda_head(torch.cat([intermittency_repr, cross_repr, lambda_raw_repr], dim=1))
        return torch.cat([lambda_raw, h_raw], dim=1)


class LambdaCurvatureDecoder(nn.Module):
    def __init__(
        self,
        moment_qs: tuple[float, ...],
        num_scales: int,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        q = torch.tensor(moment_qs, dtype=torch.float32)
        design = torch.stack([q, -0.5 * q * (q - 2.0)], dim=1)
        self.register_buffer("parabolic_pinv", torch.linalg.pinv(design))
        self.high_q_start = next(i for i, q_val in enumerate(moment_qs) if q_val >= 1.5)

        feature_dim = len(moment_qs) * 2
        feature_dim += (len(moment_qs) - self.high_q_start) * 2
        feature_dim += 6
        feature_dim += 3
        feature_dim += 2
        feature_dim += 4
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        zeta_proxy = stats["zeta_proxy"]
        q_curvature = stats["q_curvature"]
        slope_high_q = stats["slope_surface"][:, :, self.high_q_start:]
        high_q_mean = slope_high_q.mean(dim=1)
        high_q_std = slope_high_q.std(dim=1)

        wavelet_cumulants = stats["wavelet_cumulants"]
        cumulant_mean = wavelet_cumulants.mean(dim=1)
        cumulant_slope = torch.sum(
            (wavelet_cumulants - wavelet_cumulants.mean(dim=1, keepdim=True))
            * stats["log_scales"].view(1, -1, 1),
            dim=1,
        ) / torch.sum(stats["log_scales"].square()).clamp_min(1e-6)

        corr = stats["cross_scale_corr"]
        eye = torch.eye(corr.shape[1], dtype=torch.bool, device=corr.device).unsqueeze(0).expand(corr.shape[0], -1, -1)
        offdiag = corr.masked_select(~eye).view(corr.shape[0], -1)
        corr_summary = torch.stack(
            [offdiag.mean(dim=1), offdiag.std(dim=1), offdiag.abs().max(dim=1).values],
            dim=1,
        )

        parabolic_coef = zeta_proxy @ self.parabolic_pinv.T
        zeta_fit = parabolic_coef @ torch.linalg.pinv(self.parabolic_pinv).T
        residual = zeta_proxy - zeta_fit
        residual_summary = torch.stack(
            [
                residual.abs().mean(dim=1),
                residual.std(dim=1),
                residual[:, self.high_q_start:].abs().mean(dim=1),
                residual[:, self.high_q_start:].std(dim=1),
            ],
            dim=1,
        )

        features = torch.cat(
            [
                zeta_proxy,
                q_curvature,
                high_q_mean,
                high_q_std,
                cumulant_mean,
                cumulant_slope,
                corr_summary,
                parabolic_coef,
                residual_summary,
            ],
            dim=1,
        )
        return self.net(features)


class PCSMINV3Regressor(PCSMINV2Regressor):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        hidden_dim: int = 160,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            scales=scales,
            moment_qs=moment_qs,
            hidden_dim=hidden_dim,
            use_raw_gates=False,
        )
        self.lambda_curvature_decoder = LambdaCurvatureDecoder(
            moment_qs=moment_qs,
            num_scales=len(scales),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.frontend(x)
        stats["log_scales"] = self.frontend.centered_log_scales
        roughness_repr = self.roughness_branch(self._roughness_features(stats))
        intermittency_repr = self.intermittency_branch(self._intermittency_features(stats))

        cross_corr = stats["cross_scale_corr"]
        wavelet_cumulants = stats["wavelet_cumulants"]
        cross_repr = self.cross_scale_branch(cross_corr, wavelet_cumulants, self.frontend.log_scales)

        raw_repr = self._raw_features(x)
        h_raw = self.h_head(torch.cat([roughness_repr, raw_repr], dim=1))
        lambda_raw = self.lambda_head(torch.cat([intermittency_repr, cross_repr, raw_repr], dim=1))
        lambda_raw = lambda_raw + self.lambda_curvature_decoder(stats)
        return torch.cat([lambda_raw, h_raw], dim=1)


def _batch_standardized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target_detached = target.detach()
    del eps
    std = target_detached.std().clamp_min(1.0)
    return F.smooth_l1_loss(pred / std, target_detached / std)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def _gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, scale)


class LMMINetRegressor(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        hidden_dim: int = 160,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        if output_dim != 2:
            raise ValueError("LMMINetRegressor currently supports parameter mode with output_dim=2.")

        self.frontend = MRWStatisticalFrontend(scales=scales, moment_qs=moment_qs)
        self.num_q = len(moment_qs)
        self.num_scales = len(scales)
        self.num_corr_offdiag = self.num_scales * (self.num_scales - 1)
        self.high_q_start = next(i for i, q in enumerate(moment_qs) if q >= 1.5)
        self._last_aux: dict[str, torch.Tensor] = {}

        raw_channels = 48
        self.raw_stem = nn.Sequential(
            nn.Conv1d(1, raw_channels, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(raw_channels),
            nn.GELU(),
        )
        self.raw_tcn = nn.Sequential(
            TCNBlock(raw_channels, dilation=1, dropout=dropout),
            TCNBlock(raw_channels, dilation=2, dropout=dropout),
            TCNBlock(raw_channels, dilation=4, dropout=dropout),
            TCNBlock(raw_channels, dilation=8, dropout=dropout),
            TCNBlock(raw_channels, dilation=16, dropout=dropout),
        )
        self.raw_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(raw_channels, hidden_dim),
            nn.GELU(),
        )

        self.dependency_encoder = CrossScaleGraphAttention(
            num_scales=len(scales),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        slope_feature_dim = 7 + self.num_scales * 2 + 6
        curvature_feature_dim = self.num_q * 2 + (self.num_q - self.high_q_start) * 2 + 6

        self.slope_encoder = nn.Sequential(
            nn.LayerNorm(slope_feature_dim + hidden_dim),
            nn.Linear(slope_feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.curvature_encoder = nn.Sequential(
            nn.LayerNorm(curvature_feature_dim + hidden_dim),
            nn.Linear(curvature_feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.dependency_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        self.h_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 3),
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lambda_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 3),
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.slope_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_q),
        )
        self.curvature_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_q),
        )
        self.dependency_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_corr_offdiag),
        )

    def _raw_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.raw_stem(x.unsqueeze(1))
        y = self.raw_tcn(y)
        return self.raw_projection(y)

    def _slope_features(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        zeta_proxy = stats["zeta_proxy"]
        wavelet_cumulants = stats["wavelet_cumulants"]
        q1_slope = stats["slope_surface"][:, :, 2]
        q2_slope = stats["slope_surface"][:, :, 6]
        cumulant_mean = wavelet_cumulants.mean(dim=1)
        cumulant_slope = torch.sum(
            (wavelet_cumulants - wavelet_cumulants.mean(dim=1, keepdim=True))
            * self.frontend.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.frontend.scale_slope_denom
        return torch.cat([zeta_proxy[:, :7], q1_slope, q2_slope, cumulant_mean, cumulant_slope], dim=1)

    def _curvature_features(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        slope_high_q = stats["slope_surface"][:, :, self.high_q_start:]
        high_q_mean = slope_high_q.mean(dim=1)
        high_q_std = slope_high_q.std(dim=1)
        wavelet_cumulants = stats["wavelet_cumulants"]
        cumulant_mean = wavelet_cumulants.mean(dim=1)
        cumulant_slope = torch.sum(
            (wavelet_cumulants - wavelet_cumulants.mean(dim=1, keepdim=True))
            * self.frontend.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.frontend.scale_slope_denom
        return torch.cat(
            [stats["zeta_proxy"], stats["q_curvature"], high_q_mean, high_q_std, cumulant_mean, cumulant_slope],
            dim=1,
        )

    def _orthogonality_loss(self, *latents: torch.Tensor) -> torch.Tensor:
        loss = latents[0].new_zeros(())
        count = 0
        normalized = [F.normalize(z, dim=1) for z in latents]
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                loss = loss + (normalized[i] * normalized[j]).sum(dim=1).square().mean()
                count += 1
        return loss / max(count, 1)

    def mechanism_auxiliary_loss(self) -> torch.Tensor:
        return self._last_aux["reconstruction_loss"] + 0.05 * self._last_aux["orthogonality_loss"]

    def latent_diagnostics(self) -> dict[str, torch.Tensor]:
        return self._last_aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.frontend(x)
        raw = self._raw_features(x)
        dependency_repr = self.dependency_encoder(
            stats["cross_scale_corr"],
            stats["wavelet_cumulants"],
            self.frontend.log_scales,
        )

        z_slope = self.slope_encoder(torch.cat([self._slope_features(stats), raw], dim=1))
        z_curvature = self.curvature_encoder(torch.cat([self._curvature_features(stats), dependency_repr], dim=1))
        z_dependency = self.dependency_latent(dependency_repr)

        h_raw = self.h_head(torch.cat([z_slope, z_dependency, z_curvature.detach() * 0.0], dim=1))
        lambda_raw = self.lambda_head(torch.cat([z_curvature, z_dependency, z_slope.detach() * 0.0], dim=1))

        eye = torch.eye(self.num_scales, dtype=torch.bool, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        corr_target = stats["cross_scale_corr"].masked_select(~eye).view(x.shape[0], -1)
        recon_loss = (
            _batch_standardized_mse(self.slope_decoder(z_slope), stats["zeta_proxy"])
            + _batch_standardized_mse(self.curvature_decoder(z_curvature), stats["q_curvature"])
            + _batch_standardized_mse(self.dependency_decoder(z_dependency), corr_target)
        )
        orth_loss = self._orthogonality_loss(z_slope, z_curvature, z_dependency)
        self._last_aux = {
            "reconstruction_loss": recon_loss,
            "orthogonality_loss": orth_loss,
            "z_slope_train": z_slope,
            "z_curvature_train": z_curvature,
            "z_dependency_train": z_dependency,
            "z_slope": z_slope.detach(),
            "z_curvature": z_curvature.detach(),
            "z_dependency": z_dependency.detach(),
        }
        return torch.cat([lambda_raw, h_raw], dim=1)


class LMMICurvatureRegressor(LMMINetRegressor):
    """LMMI variant that makes intermittency recovery a first-class mechanism.

    The lambda2 path is anchored on explicit curvature statistics rather than a
    fully free mixed latent head. Cross-scale dependency is kept as a small
    zero-initialized residual so the model can correct finite-sample effects
    without bypassing the curvature mechanism.
    """

    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.1,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        moment_qs: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0),
        hidden_dim: int = 160,
        latent_dim: int = 64,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            scales=scales,
            moment_qs=moment_qs,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        q = torch.tensor(moment_qs, dtype=torch.float32)
        design = torch.stack([q, -0.5 * q * (q - 2.0)], dim=1)
        self.register_buffer("curvature_parabolic_pinv", torch.linalg.pinv(design))

        high_q_count = len(moment_qs) - self.high_q_start
        curvature_anchor_dim = self.num_q + high_q_count * 2 + 6 + 3 + 2 + 4
        self.curvature_anchor_encoder = nn.Sequential(
            nn.LayerNorm(curvature_anchor_dim),
            nn.Linear(curvature_anchor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.lambda_anchor_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lambda_dependency_residual = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.lambda_dependency_residual[-1].weight)
        nn.init.zeros_(self.lambda_dependency_residual[-1].bias)

        self.curvature_lambda_probe = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.curvature_h_adversary = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _curvature_anchor_features(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        slope_high_q = stats["slope_surface"][:, :, self.high_q_start:]
        high_q_mean = slope_high_q.mean(dim=1)
        high_q_std = slope_high_q.std(dim=1)

        wavelet_cumulants = stats["wavelet_cumulants"]
        cumulant_mean = wavelet_cumulants.mean(dim=1)
        cumulant_slope = torch.sum(
            (wavelet_cumulants - wavelet_cumulants.mean(dim=1, keepdim=True))
            * self.frontend.centered_log_scales.view(1, -1, 1),
            dim=1,
        ) / self.frontend.scale_slope_denom

        corr = stats["cross_scale_corr"]
        eye = torch.eye(self.num_scales, dtype=torch.bool, device=corr.device).unsqueeze(0).expand(corr.shape[0], -1, -1)
        offdiag = corr.masked_select(~eye).view(corr.shape[0], -1)
        corr_summary = torch.stack(
            [offdiag.mean(dim=1), offdiag.std(dim=1), offdiag.abs().max(dim=1).values],
            dim=1,
        )

        parabolic_coef = stats["zeta_proxy"] @ self.curvature_parabolic_pinv.T
        zeta_fit = parabolic_coef @ torch.linalg.pinv(self.curvature_parabolic_pinv).T
        residual = stats["zeta_proxy"] - zeta_fit
        residual_high_q = residual[:, self.high_q_start:]
        residual_summary = torch.stack(
            [
                residual.abs().mean(dim=1),
                residual.std(dim=1),
                residual_high_q.abs().mean(dim=1),
                residual_high_q.std(dim=1),
            ],
            dim=1,
        )

        return torch.cat(
            [
                stats["q_curvature"],
                high_q_mean,
                high_q_std,
                cumulant_mean,
                cumulant_slope,
                corr_summary,
                parabolic_coef,
                residual_summary,
            ],
            dim=1,
        )

    def supervised_auxiliary_loss(self, target_norm: torch.Tensor) -> torch.Tensor:
        aux = self._last_aux
        lambda_target = target_norm[:, 0:1]
        h_target = target_norm[:, 1:2]
        lambda_probe = aux["lambda_anchor_probe"]
        h_adv = aux["curvature_h_adversarial_probe"]
        return 0.35 * F.mse_loss(lambda_probe, lambda_target) + 0.03 * F.mse_loss(h_adv, h_target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.frontend(x)
        raw = self._raw_features(x)
        dependency_repr = self.dependency_encoder(
            stats["cross_scale_corr"],
            stats["wavelet_cumulants"],
            self.frontend.log_scales,
        )

        z_slope = self.slope_encoder(torch.cat([self._slope_features(stats), raw], dim=1))
        z_curvature = self.curvature_encoder(torch.cat([self._curvature_features(stats), dependency_repr], dim=1))
        z_dependency = self.dependency_latent(dependency_repr)
        z_curvature_anchor = self.curvature_anchor_encoder(self._curvature_anchor_features(stats))

        h_raw = self.h_head(torch.cat([z_slope, z_dependency, z_curvature.detach() * 0.0], dim=1))
        lambda_anchor = self.lambda_anchor_head(torch.cat([z_curvature, z_curvature_anchor], dim=1))
        lambda_residual = self.lambda_dependency_residual(z_dependency)
        lambda_raw = lambda_anchor + 0.25 * lambda_residual

        lambda_probe = self.curvature_lambda_probe(z_curvature)
        h_adv_probe = self.curvature_h_adversary(_gradient_reverse(z_curvature, scale=0.4))

        eye = torch.eye(self.num_scales, dtype=torch.bool, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        corr_target = stats["cross_scale_corr"].masked_select(~eye).view(x.shape[0], -1)
        recon_loss = (
            _batch_standardized_mse(self.slope_decoder(z_slope), stats["zeta_proxy"])
            + _batch_standardized_mse(self.curvature_decoder(z_curvature), stats["q_curvature"])
            + _batch_standardized_mse(self.dependency_decoder(z_dependency), corr_target)
        )
        orth_loss = self._orthogonality_loss(z_slope, z_curvature, z_dependency)
        self._last_aux = {
            "reconstruction_loss": recon_loss,
            "orthogonality_loss": orth_loss,
            "z_slope_train": z_slope,
            "z_curvature_train": z_curvature,
            "z_dependency_train": z_dependency,
            "z_slope": z_slope.detach(),
            "z_curvature": z_curvature.detach(),
            "z_dependency": z_dependency.detach(),
            "lambda_anchor_probe": lambda_probe,
            "curvature_h_adversarial_probe": h_adv_probe,
        }
        return torch.cat([lambda_raw, h_raw], dim=1)
