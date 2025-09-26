import os
import pytest
from breedspotter.infer import Predictor

@pytest.mark.skipif(not os.path.exists("checkpoints/best.pt"),
                    reason="no checkpoint available")
def test_predictor_loads():
    p = Predictor("checkpoints/best.pt")
    assert len(p.classes) > 0
