from fractions import Fraction

from fab_agent.domain.design import FabricationDesign, ObservedComponent
from fab_agent.domain.takeoff import compute_takeoff
from fab_agent.domain.validation import validate_design
from fab_agent.infrastructure.artifacts import generate_artifacts
from fab_agent.infrastructure.catalogs import CatalogBundle


def test_generates_png_svg_csv_report_and_step(
    valid_design: FabricationDesign, catalogs: CatalogBundle
) -> None:
    valid_design.spools[0].features[0].kind = "start"
    valid_design.spools[0].features[-1].kind = "end"
    valid_design.observed_components = [
        ObservedComponent(description_raw="2 - 4 inch couplings", quantity=2, kind="Coupling")
    ]
    validation = validate_design(
        valid_design,
        tolerance=Fraction(1, 16),
        catalogs=catalogs,
        allow_demo=True,
    )
    takeoff = compute_takeoff(valid_design, validation)
    artifacts = generate_artifacts(
        valid_design,
        validation,
        takeoff,
        catalogs,
        allow_demo=True,
        cad_enabled=True,
    )
    assert artifacts.files["spool-001_step"].stat().st_size > 0
    assert artifacts.files["spool-001_png"].stat().st_size > 0
    assert artifacts.files["spool-001_svg"].stat().st_size > 0
    bom = artifacts.files["bom"].read_text()
    review = artifacts.files["review"].read_text()
    assert "REVIEW OUTPUT" in bom
    assert "observed_parts_list,coupling,2" in bom
    assert "SYNTHETIC GEOMETRY" in review
    assert "they are not placed in CAD" in review
