"""Default Matplotlib reports: one atomic multipage PDF per workflow."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@contextmanager
def pdf_report(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".pdf")
    os.close(fd)
    try:
        with plt.rc_context(matplotlib.rcParamsDefault):
            with PdfPages(temporary) as pdf:
                yield pdf
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_page(pdf, figure, title):
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, .94))
    try:
        pdf.savefig(figure)
    finally:
        plt.close(figure)
