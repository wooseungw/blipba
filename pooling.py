def _get_2dPool(self, feature: torch.FloatTensor, stride: int = 2):
        """ViT patch token을 2‑D grid 로 reshape 후 pooling."""
        # feature: (B, N, d_v)
        num_frames, num_tokens, num_dim = feature.shape
        side = int(math.sqrt(num_tokens))
        assert side * side == num_tokens, "토큰 수가 완전한 정사각형이 아님"
        print(f"feature shape: {feature.shape}, side: {side}")
        feature = feature.view(num_frames, side, side, num_dim)  # (B, H, W, d_v)

        # (Optional) temporal pooling (영상 입력 대응)
        temporal_pooling = getattr(self.config, "temporal_pooling", 1)
        if temporal_pooling > 1:
            feature = feature.reshape(num_frames, num_tokens, num_dim)
            feature = feature.permute(1, 2, 0)  # (N, d_v, B)
            feature = nn.functional.avg_pool1d(feature, kernel_size=temporal_pooling, stride=temporal_pooling)
            feature = feature.permute(2, 0, 1)
            num_frames //= temporal_pooling
            feature = feature.view(num_frames, side, side, num_dim)

        # (B, H, W, d_v) → (B, d_v, H, W)
        feature = feature.permute(0, 3, 1, 2).contiguous()

        mode = getattr(self.config, "mm_spatial_pool_mode", "average")
        if mode == "average":
            feature = nn.functional.avg_pool2d(feature, stride)
        elif mode == "max":
            feature = nn.functional.max_pool2d(feature, stride)
        elif mode == "bilinear":
            height, width = feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            feature = nn.functional.interpolate(feature, size=scaled_shape, mode="bilinear")
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {mode}")
        print(f"feature shape after pooling: {feature.shape}")
        # (B, d_v, H', W') → (B, H'*W', d_v)
        feature = feature.permute(0, 2, 3, 1).contiguous().view(num_frames, -1, num_dim)
        return feature