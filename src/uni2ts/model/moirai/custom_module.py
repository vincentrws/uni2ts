from uni2ts.model.moirai import MoiraiModule
from uni2ts.module.packed_scaler import OHLCVPackedScaler

class OHLCVMoiraiModule(MoiraiModule):
    """
    Custom Moirai Module that uses OHLCVPackedScaler for normalization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scaling:
            # Configure scaler to match OHLCVLoader packing:
            # 0:Open, 1:High, 2:Low, 3:Close, 4:Vol, 5:Min, 6:Dow
            self.scaler = OHLCVPackedScaler(
                open_idx=0,
                high_idx=1,
                low_idx=2,
                volume_idx=4,     # Volume is at index 4
                minutes_idx=5,    # Minutes is at index 5
                day_of_week_idx=6 # Dow is at index 6
            )
