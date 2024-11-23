# gradio_app.py

import gradio as gr
from pathlib import Path
import os
from typing import List
import pandas as pd
import torch



from utils import (
    PDFProcessor,
    ModelProcessor,
    MaskExtractor,
    AnnotationProcessor,
    ImageProcessor,
    TabularProcessor,
    PDFConfig,
    ModelConfig,
    MaskExtractionConfig,
    AnnotationConfig,
    TabularConfig,
    SecondStepProcessor,
    SecondStepConfig,
    ExportProcessor,
    ExportConfig
)

class App:
    """Main application class for the PyPotteryLens project"""
    
    # In new_app.py, update the __init__ method

    def __init__(self):
        # Setup directories
        self.root_dir = Path(".")
        self.pred_output_dir = self.root_dir / "outputs"
        self.pdfimg_output_dir = self.root_dir / "pdf2img_outputs"
        self.models_dir = self.root_dir / "models_vision"
        self.models_classifier_dir = self.root_dir / "models_classifier"
        ##
        self.assets_dir = self.root_dir / "imgs"
        
        # Create necessary directories
        os.makedirs(self.pdfimg_output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.pred_output_dir, exist_ok=True)
        os.makedirs(self.models_classifier_dir, exist_ok=True)
        
        # Initialize processors
        self.pdf_processor = PDFProcessor(PDFConfig(output_dir=self.pdfimg_output_dir))
        
        self.model_processor = ModelProcessor(ModelConfig(
            models_dir=self.models_dir,
            pred_output_dir=self.pred_output_dir
        ))

        # Initialize mask extractor with correct paths
        mask_config = MaskExtractionConfig(
            pdfimg_output_dir=self.pdfimg_output_dir,
            pred_output_dir=self.pred_output_dir
        )
        self.mask_extractor = MaskExtractor(mask_config)

        self.annotation_processor = AnnotationProcessor(AnnotationConfig(
            pred_output_dir=self.pred_output_dir
        ))
        
        self.image_processor = ImageProcessor(
            pdfimg_output_dir=self.pdfimg_output_dir,
            pred_output_dir=self.pred_output_dir
        )

        self.tabular_processor = TabularProcessor(TabularConfig(
            pdfimg_output_dir=self.pdfimg_output_dir,
            pred_output_dir=self.pred_output_dir
        ))

        # Initialize second step processor with proper model path
        second_step_config = SecondStepConfig(
            pred_output_dir=self.pred_output_dir,
            model_path=self.models_classifier_dir / "model_classifier.pth"
        )
        self.second_step_processor = SecondStepProcessor(second_step_config)

        self.export_processor = ExportProcessor(ExportConfig(
            pred_output_dir=self.pred_output_dir,
            #export_dir=self.root_dir / "exports"
        ))

    def get_image_folders(self) -> List[str]:
        """Get list of image folders"""
        return os.listdir(self.pdfimg_output_dir)

    def get_models_list(self) -> List[str]:
        """Get list of available models"""
        return os.listdir(self.models_dir)

    def get_results_folders(self) -> List[str]:
        """Get list of result folders"""
        folder_list = os.listdir(self.pred_output_dir)
        return [folder for folder in folder_list if folder.endswith('_card')]

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface"""
        with gr.Blocks() as demo:
            self._create_header()
            
            with gr.Tabs() as tabs:
                pdf_tab = self._create_pdf_tab()
                model_tab = self._create_model_tab()
                annotation_tab = self._create_annotation_tab()
                tabular_tab = self._create_tabular_tab()
                second_step_tab = self._create_second_step_tab()
                # Refresh all dropdowns when switching tabs

                tabs.select(
                    fn=self._refresh_dropdowns,
                    inputs=None,
                    outputs=[
                        model_tab["folder_dropdown"],      # From model tab
                        model_tab["model_dropdown"],       # From model tab
                        annotation_tab["folder_dropdown"], # From annotation tab
                        tabular_tab["folder_dropdown"],    # From tabular tab
                        second_step_tab["folder_dropdown"] # From second step tab
                    ]
                )
            
            return demo
        
    def _refresh_dropdowns(self):
        """Refresh all dropdowns with current folder/model lists"""
        return [
            gr.update(choices=self.get_image_folders(), value=None),     # For model tab folder dropdown
            gr.update(choices=self.get_models_list(), value=None),       # For model tab model dropdown
            gr.update(choices=self.get_image_folders(), value=None),     # For annotation tab folder dropdown
            gr.update(choices=self.tabular_processor.get_results_folders(), value=None),  # For tabular tab dropdown
            gr.update(choices=[f for f in self.get_results_folders() if not f.endswith('transformed_card')], value=None)    # For second step tab folder dropdown
        ]

    def _create_header(self):
        """Create application header"""
        # Convert image to base64 to embed it directly in HTML
        image_path = os.path.join(os.path.dirname(__file__), "imgs", "pypotterylens.png")
        with open(image_path, "rb") as img_file:
            import base64
            img_data = base64.b64encode(img_file.read()).decode()

        return gr.HTML(f"""
            <div style="display: flex; align-items: center; gap: 20px;">
                <img src="data:image/png;base64,{img_data}" 
                    alt="pottery icon" 
                    style="border-radius: 8px; width: 100px;"/>
                <div>
                    <h1>PyPotteryLens</h1>
                    <span>Archaeological Pottery Documentation Tool 
                        <span style="font-size: 0.9em; color: #666;">v0.1.0</span>
                    </span>
                </div>
            </div>
        """)

    def _create_pdf_tab(self):
        """Create PDF processing tab"""
        with gr.Tab("PDF document processing"):
            gr.HTML("""
                <h1>Select a PDF file</h1>
                It will be converted to JPG format
            """)
            
            with gr.Row():
                upload_button = gr.UploadButton(
                    "Click to Upload and process a File",
                    file_types=[".pdf"],
                    file_count="single"
                )
            
            with gr.Row():
                with gr.Column():
                    split_pages = gr.Checkbox(
                        label="Split scanned pages",
                        value=False,
                        info="Check this if each PDF page contains two actual pages (left and right)"
                    )
            
            file_name = gr.Text(
                label="File Name",
                info="Selected PDF file path",
                interactive=False
            )
            
            # Modified upload handler to include split_pages parameter
            upload_button.upload(
                fn=self.pdf_processor.process_pdf,
                inputs=[
                    upload_button,
                    split_pages
                ],
                outputs=file_name
            )

            gr.HTML("""
                <div style="margin-top: 10px; padding: 10px; border-radius: 5px;">
                    <p><strong>Note:</strong></p>
                    <p>Use "Split scanned pages" when:</p>
                    <ul>
                        <li>Your PDF contains scanned books or documents</li>
                        <li>Each PDF page shows two actual pages (left and right)</li>
                        <li>You want to process each page separately</li>
                    </ul>
                </div>
            """)

    def _create_model_tab(self):
        """Create model application tab"""
        with gr.Tab("Apply Model") as model_tab:
            with gr.Row():
                # Left side - Controls
                with gr.Column(scale=1):
                    # Model Selection Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                📁 Input Selection
                            </h3>
                        """)
                        folder_dropdown = gr.Dropdown(
                            label="Image Folder",
                            choices=self.get_image_folders(),
                            interactive=True,
                            info="Select the folder containing your images"
                        )
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=self.get_models_list(),
                            interactive=True,
                            info="Select the YOLO model to apply"
                        )
                        gr.HTML("</div>")
                    
                    # Model Parameters Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                ⚙️ Model Parameters
                            </h3>
                        """)
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Confidence Threshold",
                            info="Lower values detect more objects but may increase false positives"
                        )
                        
                        with gr.Row():
                            kernel_number = gr.Number(
                                label="Kernel Size",
                                value=2,
                                minimum=1,
                                maximum=10,
                                step=1,
                                interactive=True,
                                info="Size of the processing kernel"
                            )
                            iterations_number = gr.Number(
                                label="Iterations",
                                value=10,
                                minimum=1,
                                maximum=50,
                                step=1,
                                interactive=True,
                                info="Number of processing iterations"
                            )
                        gr.HTML("</div>")
                    
                    # Advanced Options Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                🔧 Advanced Options
                            </h3>
                        """)
                        diagnostic_checkbox = gr.Checkbox(
                            label="Diagnostic Mode",
                            value=False,
                            info="Process only first 25 images for testing"
                        )
                        gr.HTML("</div>")
                    
                    # Process Button Section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                        """)
                        process_button = gr.Button(
                            value="🚀 Apply Model",
                            variant="primary",
                            scale=1
                        )
                        status_text = gr.Text(
                            label="Status",
                            placeholder="Ready to process...",
                            interactive=False
                        )
                        gr.HTML("</div>")

                # Right side - Preview
                with gr.Column(scale=1):
                    empty_msg = gr.HTML(
                        """
                        <div style="display: flex; justify-content: center; align-items: center; 
                                height: 400px; background-color: #f8f9fa; border-radius: 8px;
                                border: 2px dashed #dee2e6;">
                            <div style="text-align: center; color: #6c757d;">
                                <h3>📁 No folder selected</h3>
                                <p>Select an image folder to preview its contents</p>
                            </div>
                        </div>
                        """,
                        visible=True
                    )
                    
                    gallery = gr.Gallery(
                        label="Images in selected folder",
                        show_label=True,
                        columns=4,
                        visible=False
                    )

                # Event handlers
                model_tab.select(self.get_image_folders, outputs=folder_dropdown)
                model_tab.select(self.get_models_list, outputs=model_dropdown)
                
                def update_gallery(folder):
                    if not folder:
                        return {
                            empty_msg: gr.update(visible=True),
                            gallery: gr.update(visible=False, value=None)
                        }
                    images = self.image_processor.return_images(folder)
                    return {
                        empty_msg: gr.update(visible=False),
                        gallery: gr.update(visible=True, value=images)
                    }
                
                folder_dropdown.change(
                    fn=update_gallery,
                    inputs=folder_dropdown,
                    outputs=[empty_msg, gallery]
                )

                process_button.click(
                    fn=self.model_processor.apply_model,
                    inputs=[
                        folder_dropdown,
                        model_dropdown,
                        confidence_slider,
                        diagnostic_checkbox,
                        kernel_number,
                        iterations_number
                    ],
                    outputs=status_text
                )

        return {
            "folder_dropdown": folder_dropdown,
            "model_dropdown": model_dropdown
        }

    def _create_annotation_tab(self):
        """Create annotation review tab with properly displayed editor"""
        # Add the CSS for FileExplorer fix first
        with gr.Tab("Review Annotations and extract masks") as annotation_tab:
            gr.HTML("""
                <style>
                    .file-explorer {
                        overflow: hidden !important;
                    }
                    .file-explorer > div {
                        height: 95% !important;
                    }
                </style>
            """)
            
            with gr.Row():
                # Left side - Controls
                with gr.Column(scale=1):
                    # Image Selection section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                📁 Image Selection
                            </h3>
                        """)
                        folder_dropdown = gr.Dropdown(
                            label="Select Folder",
                            choices=self.get_image_folders(),
                            interactive=True,
                            info="Choose the folder containing images to annotate"
                        )
                        file_explorer = gr.FileExplorer(
                            visible=False,
                            height=300, #300
                            elem_classes=["file-explorer"]
                        )
                        gr.HTML("</div>")

                    # Size Control section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                📐 Editor Size
                            </h3>
                        """)
                        size_slider = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Editor Size (%)",
                            info="Adjust the size of the editor"
                        )
                        gr.HTML("</div>")

                    # Extraction section
                    with gr.Group():
                        gr.HTML("""
                            <div style="padding: 1em; border-radius: 8px;>
                            <h3 style="margin-bottom: 1em">
                                🎯 Extraction
                            </h3>
                        """)
                        extract_button = gr.Button(
                            "📤 Extract Masks",
                            variant="primary"
                        )
                        status_text = gr.Text(
                            label="Status",
                            placeholder="Ready to extract masks..."
                        )
                        gr.HTML("</div>")

                # Right side - Editor
                with gr.Column(scale=2):
                    empty_editor_msg = gr.HTML(
                        """
                        <div style="display: flex; justify-content: center; align-items: center; 
                                height: 400px; background-color: #f8f9fa; border-radius: 8px;
                                border: 2px dashed #dee2e6;">
                            <div style="text-align: center; color: #6c757d;">
                                <h3>🖼️ No Image Selected</h3>
                                <p>Select a folder and image to start annotating</p>
                            </div>
                        </div>
                        """,
                        visible=True
                    )
                    
                    image_editor = gr.ImageEditor(
                        interactive=True,
                        layers=False,
                        sources=[],
                        transforms=[],
                        brush=gr.Brush(colors=["#80808080"], default_size=20),  # Hex color with alpha
                        eraser=gr.Eraser(default_size=20),
                        visible=True,
                        height="50%",
                        width="50%"
                    )

            # Event Handlers
            def update_file_explorer(folder):
                if not folder:
                    return {
                        file_explorer: gr.update(visible=False),
                        empty_editor_msg: gr.update(visible=True),
                        image_editor: gr.update(visible=False)
                    }
                return {
                    file_explorer: self.annotation_processor.get_file_explorer(folder),
                    empty_editor_msg: gr.update(visible=True),
                    image_editor: gr.update(visible=False)
                }

            def update_image_editor(file_path):
                if not file_path:
                    return {
                        empty_editor_msg: gr.update(visible=True),
                        image_editor: gr.update(visible=False)
                    }
                result = self.annotation_processor.file_selection(file_path)
                return {
                    empty_editor_msg: gr.update(visible=False),
                    image_editor: gr.update(visible=True, value=result)
                }

            def update_editor_size(size):
                return gr.update(width=f"{size}%", height=f"{size}%")

            # Connect events
            annotation_tab.select(self.get_image_folders, outputs=folder_dropdown)
            
            folder_dropdown.change(
                fn=update_file_explorer,
                inputs=folder_dropdown,
                outputs=[file_explorer, empty_editor_msg, image_editor]
            )

            file_explorer.change(
                fn=update_image_editor,
                inputs=file_explorer,
                outputs=[empty_editor_msg, image_editor]
            )

            image_editor.change(
                fn=self.annotation_processor.save_annotation,
                inputs=[folder_dropdown, image_editor, file_explorer]
            )

            extract_button.click(
                fn=self.mask_extractor.extract_masks,
                inputs=[folder_dropdown],
                outputs=status_text
            )

            # Connect size control event
            size_slider.change(
                fn=update_editor_size,
                inputs=[size_slider],
                outputs=image_editor
            )

        return {
            "folder_dropdown": folder_dropdown
        }

    def _create_tabular_tab(self):
        """Create tabular information tab with mask-filtered navigation"""
        with gr.Tab("Tabular Information") as tabular_info:
            with gr.Row():
                gr.HTML("""
                    <div style="padding: 1em; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 1em;">
                        <h2 style="margin: 0; color: #2c3e50;">📊 Tabular Information</h2>
                        <p style="margin: 0.5em 0 0 0; color: #7f8c8d;">
                            View and edit data associated with extracted masks
                        </p>
                    </div>
                """)

            # Folder Selection
            with gr.Row():
                with gr.Column():
                    drop_txt = gr.Dropdown(
                        label="Results Folder",
                        choices=self.tabular_processor.get_results_folders(),
                        info="Select a folder to view its data",
                        scale=2
                    )

            with gr.Row():
                # Navigation Controls
                with gr.Column(scale=1):
                    with gr.Row():
                        img_num = gr.Number(
                            value=0,
                            label="Current Image",
                            show_label=True,
                            interactive=False,
                            scale=1
                        )
                        max_img = gr.Number(
                            value=0,
                            label="Total Images",
                            show_label=True,
                            interactive=False,
                            scale=1
                        )
                        img_num_bottom = gr.Number(
                            value=None,
                            label="Go to Image",
                            show_label=True,
                            interactive=True,
                            scale=1
                        )
                        img_button = gr.Button("🔍 Go") #0.5

                    with gr.Row():
                        prev_button = gr.Button("◀ Previous", scale=1)
                        next_button = gr.Button("Next ▶", scale=1)

            # Main Content Area
            with gr.Row():
                # Image Display - Larger Size
                with gr.Column(scale=3):
                    img = gr.AnnotatedImage(
                        show_label=True,
                        height=800,
                        width="100%",
                        container=True
                    )

                # Table Display and Controls
                with gr.Column(scale=2):
                    # New Column Controls
                    with gr.Row():
                        new_column_name = gr.Textbox(
                            label="New Column Name",
                            placeholder="Enter column name...",
                            scale=2
                        )
                        add_column_btn = gr.Button("➕ Add Column", scale=1)

                    # Data Table
                    table = gr.DataFrame(
                        interactive=True,
                        wrap=True,
                        column_widths="auto"
                    )

            # Event Handlers
            def get_valid_images(folder: str) -> list:
                """Get list of images that have corresponding masks"""
                if not folder:
                    return []
                
                # Get base folder name (removing '_card' suffix)
                base_folder = folder.split("_card")[0]
                mask_folder = f"{base_folder}_mask"
                
                # Get list of mask files
                mask_path = self.pred_output_dir / mask_folder
                if not mask_path.exists():
                    return []
                
                # Get mask files and extract corresponding image names
                mask_files = os.listdir(mask_path)
                valid_images = set()
                
                for mask_file in mask_files:
                    if mask_file.endswith("_mask_layer.png"):
                        # Extract original image name
                        img_name = mask_file.replace("_mask_layer.png", "")
                        valid_images.add(img_name)
                
                return sorted(list(valid_images))

            def safe_image_selection(txt, num):
                try:
                    # Get the valid images first
                    valid_images = get_valid_images(txt)
                    total_images = len(valid_images)
                    
                    if total_images == 0:
                        return None, 0, None, 0
                    
                    # Validate the image number
                    num = max(0, min(num, total_images - 1))
                    
                    # Get the specific image name
                    img_name = valid_images[num]
                    
                    # Get the results using the image name
                    result = self.tabular_processor.image_selection(txt, num)
                    return result[0], num, result[2], total_images - 1
                    
                except Exception as e:
                    print(f"Error in image selection: {str(e)}")
                    return None, 0, None, 0

            # Connect Events with updated handlers
            def handle_dropdown_select(txt, num):
                image, number, table_data, max_imgs = safe_image_selection(txt, num)
                return [image, number, table_data, max_imgs]

            drop_txt.select(
                fn=handle_dropdown_select,
                inputs=[drop_txt, img_num],
                outputs=[img, img_num, table, max_img]
            )

            def handle_next(txt, num):
                image, number, table_data, max_imgs = safe_image_selection(txt, num + 1)
                return [image, number, table_data, max_imgs]

            next_button.click(
                fn=handle_next,
                inputs=[drop_txt, img_num],
                outputs=[img, img_num, table, max_img]
            )

            def handle_prev(txt, num):
                image, number, table_data, max_imgs = safe_image_selection(txt, max(0, num - 1))
                return [image, number, table_data, max_imgs]

            prev_button.click(
                fn=handle_prev,
                inputs=[drop_txt, img_num],
                outputs=[img, img_num, table, max_img]
            )

            def handle_goto(txt, num, target):
                try:
                    target_num = int(target) if target is not None else 0
                except:
                    target_num = 0
                
                image, number, table_data, max_imgs = safe_image_selection(txt, target_num)
                return [image, number, table_data, max_imgs]

            img_button.click(
                fn=handle_goto,
                inputs=[drop_txt, img_num, img_num_bottom],
                outputs=[img, img_num, table, max_img]
            )

            # Add Column Event
            def add_new_column(table_data: pd.DataFrame, column_name: str) -> pd.DataFrame:
                if column_name and column_name not in table_data.columns:
                    table_data[column_name] = ""
                return table_data

            add_column_btn.click(
                fn=add_new_column,
                inputs=[table, new_column_name],
                outputs=table
            )

            # Auto-save changes
            table.change(
                fn=self.tabular_processor.save_table,
                inputs=[table, drop_txt],
                outputs=None
            )

            return {
                "folder_dropdown": drop_txt
            }
        

    def _create_second_step_tab(self):
        with gr.Tab("Post Processing"):
                # Input Selection Section
                with gr.Row():
                    # Left side - Controls
                    with gr.Column(scale=1):
                        # Folder Selection Section
                        with gr.Group():
                            gr.HTML("""
                                <div style="padding: 1em; border-radius: 8px;>
                                <h3 style="margin-bottom: 1em">
                                    📁 Input Selection
                                </h3>
                            """)
                            folder_dropdown = gr.Dropdown(
                                label="Results Folder",
                                choices=[f for f in self.get_results_folders() if not f.endswith('transformed_card')],
                                interactive=True,
                                info="Select folder containing extracted masks"
                            )
                            gr.HTML("</div>")
                        
                        # Model Parameters Section
                        with gr.Group():
                            gr.HTML("""
                                <div style="padding: 1em; border-radius: 8px;>
                                <h3 style="margin-bottom: 1em">
                                    ⚙️ Processing Options
                                </h3>
                            """)
                            auto_flip_vertical = gr.Checkbox(
                                label="Auto Vertical Flip",
                                value=True,
                                info="Apply vertical flipping during model processing"
                            )
                            auto_flip_horizontal = gr.Checkbox(
                                label="Auto Horizontal Flip",
                                value=True,
                                info="Apply horizontal flipping during model processing"
                            )
                            gr.HTML("</div>")
                        
               # Process Buttons Section
                        with gr.Group():
                            gr.HTML("""
                                <div style="padding: 1em; border-radius: 8px;>
                                <h3 style="margin-bottom: 1em">
                                    🚀 Actions
                                </h3>
                                """)
                            with gr.Row():
                                process_button = gr.Button(
                                    "🔍 Process All Images",
                                    variant="primary",
                                    scale=1
                                )
                                merge_button = gr.Button(
                                    "📋 Merge Annotations",
                                    variant="secondary",
                                    scale=1
                                )
                            with gr.Row():
                                export_button = gr.Button(
                                    "📦 Export Results",
                                    variant="primary",
                                    scale=1
                                )
                            status_text = gr.Text(
                                label="Status",
                                interactive=False
                            )
                            gr.HTML("</div>")
            
                        # Create export modal
                   

                    # Right side - Preview and Controls
                    with gr.Column(scale=2):
                        # Navigation Controls
                        with gr.Row():
                            prev_button = gr.Button("◀ Previous", scale=1)
                            next_button = gr.Button("Next ▶", scale=1)
                            image_counter = gr.Number(
                                value=0,
                                label="Image",
                                interactive=True,
                                scale=1
                            )
                        
                        # Images Display
                        with gr.Row():
                            # Original Image
                            with gr.Column():
                                gr.HTML("<h4 style='text-align: center;'>Original Image</h4>")
                                original_image = gr.Image(
                                    label="Original",
                                    show_label=False,
                                    height=400
                                )

                            # Transformed Image
                            with gr.Column():
                                gr.HTML("<h4 style='text-align: center;'>Processed Image</h4>")
                                transformed_image = gr.Image(
                                    label="Processed",
                                    show_label=False,
                                    height=400
                                )
                                with gr.Row():
                                        flip_vertical_btn = gr.Button("↕️ Flip Vertical", scale=1)
                                        flip_horizontal_btn = gr.Button("↔️ Flip Horizontal", scale=1)
                                        type_dropdown = gr.Dropdown(
                                        label="Type",
                                        choices=["ENT", "FRAG"],
                                        interactive=True,
                                        scale=1
                                    )
        
                export_dialog, acronym_input, export_status, cancel_btn, export_btn, pdf_export, page_size, scale_factor = self._create_export_dialog()

                def handle_export_click(folder: str):
                    """Show export dialog when export button is clicked"""
                    if not folder:
                        return [
                            gr.update(visible=False),  # export_dialog
                            "",                        # acronym_input
                            "",                        # export_status
                            False,                     # pdf_export
                            "A4",                      # page_size
                            1.0,                       # scale_factor
                            "Please select a folder first"  # status_text
                        ]
                    return [
                        gr.update(visible=True),  # export_dialog
                        "",                       # acronym_input
                        "",                       # export_status
                        False,                    # pdf_export
                        "A4",                     # page_size
                        1.0,                      # scale_factor
                        ""                        # status_text
                    ]

                def handle_export_confirm(folder: str, acronym: str, export_pdf: bool,
                                       page_size: str, scale_factor: float):
                    """Handle the export confirmation with PDF options"""
                    # Validate acronym
                    validation_msg = validate_acronym(acronym)
                    if validation_msg:
                        return [
                            validation_msg,              # export_status
                            gr.update(visible=True),     # export_dialog
                            validation_msg               # status_text
                        ]
                    
                    # Process export with PDF options
                    result = self.export_processor.export_results(
                        folder=folder,
                        acronym=acronym,
                        export_pdf=export_pdf,
                        page_size=page_size,
                        scale_factor=scale_factor
                    )
                    
                    return [
                        result,                     # export_status
                        gr.update(visible=False),   # export_dialog
                        result                      # status_text
                    ]

                # Connect event handlers
                export_button.click(
                    fn=handle_export_click,
                    inputs=[folder_dropdown],
                    outputs=[
                        export_dialog,
                        acronym_input,
                        export_status,
                        pdf_export,
                        page_size,
                        scale_factor,
                        status_text
                    ]
                )
                
                export_btn.click(
                    fn=handle_export_confirm,
                    inputs=[
                        folder_dropdown,
                        acronym_input,
                        pdf_export,
                        page_size,
                        scale_factor
                    ],
                    outputs=[
                        export_status,
                        export_dialog,
                        status_text
                    ]
                )
                
                cancel_btn.click(
                    fn=lambda: [gr.update(visible=False), "", ""],
                    inputs=None,
                    outputs=[export_dialog, export_status, status_text]
                )


        def process_with_options(folder, flip_v, flip_h):
            if not folder:
                return {
                    status_text: "Please select a folder",
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }
            
            try:
                # Update the model processor configuration with flip options
                self.second_step_processor.set_flip_options(flip_v, flip_h)
                
                # Process the folder
                results = self.second_step_processor.process_folder(folder)
                
                if results.empty:
                    return {
                        status_text: "No images were processed successfully",
                        original_image: None,
                        transformed_image: None,
                        type_dropdown: None,
                        image_counter: 0
                    }
                
                # Load first processed image
                first_row = results.iloc[0]
                original_path = self.second_step_processor.get_original_path(folder, first_row['filename'])
                transformed_path = self.second_step_processor.get_transformed_path(folder, first_row['filename'])
                
                return {
                    status_text: f"Successfully processed {len(results)} images",
                    original_image: str(original_path),
                    transformed_image: str(transformed_path),
                    type_dropdown: first_row['type'],
                    image_counter: 0
                }
                    
            except Exception as e:
                error_msg = f"Error processing folder: {str(e)}"
                print(error_msg)
                return {
                    status_text: error_msg,
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }

        def manual_flip(folder, image_idx, flip_type):
            """Handle manual image flipping"""
            if not folder or image_idx is None:
                return None, None, "No image selected"
                
            try:
                results = self.second_step_processor.load_results(folder)
                if results.empty or image_idx >= len(results):
                    return None, None, "Invalid image index"
                    
                filename = results.iloc[image_idx]['filename']
                
                # Perform the flip operation
                flipped = self.second_step_processor.manual_flip(
                    folder, filename, flip_type
                )
                
                if flipped is None:
                    return None, None, "Error flipping image"
                
                # Get paths for display
                original_path = self.second_step_processor.get_original_path(folder, filename)
                processed_path = self.second_step_processor.get_transformed_path(folder, filename)
                
                return str(original_path), str(processed_path), "Image flipped successfully"
                
            except Exception as e:
                return None, None, f"Error during flip: {str(e)}"

        def navigate_images(folder, direction, current_idx):
            """Handle image navigation"""
            if not folder:
                return {
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: current_idx,
                    status_text: "No folder selected"
                }
                
            results = self.second_step_processor.load_results(folder)
            if results.empty:
                return {
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: current_idx,
                    status_text: "No results found"
                }
                
            # Calculate new index
            new_idx = current_idx + (1 if direction == "next" else -1)
            new_idx = max(0, min(new_idx, len(results) - 1))
            
            # Load new image
            row = results.iloc[new_idx]
            original_path = self.second_step_processor.get_original_path(folder, row['filename'])
            transformed_path = self.second_step_processor.get_transformed_path(folder, row['filename'])
            
            return {
                original_image: str(original_path),
                transformed_image: str(transformed_path),
                type_dropdown: row['type'],
                image_counter: new_idx,
                status_text: f"Image {new_idx + 1} of {len(results)}"
            }

        def update_type(folder, image_idx, new_type):
            """Handle type update"""
            if not folder or image_idx is None:
                return "No image selected"
            
            try:
                results = self.second_step_processor.load_results(folder)
                if results.empty or image_idx >= len(results):
                    return "Invalid image index"
                    
                filename = results.iloc[image_idx]['filename']
                self.second_step_processor.update_result(folder, filename, {'type': new_type})
                return f"Updated type to {new_type}"
                
            except Exception as e:
                return f"Error updating type: {str(e)}"
            
        def merge_annotations(folder):
            if not folder:
                return "Please select a folder first"
            
            try:
                # Get paths - mask_info.csv is in the _card folder
                annots_path = self.pred_output_dir / folder / "mask_info.csv"
                results_path = self.second_step_processor.get_transformed_folder_path(folder) / "classifications.csv"
                
                if not annots_path.exists():
                    return f"Annotations file not found at {annots_path}"
                if not results_path.exists():
                    return "Classifications file not found. Process images first."
                    
                # Load CSVs
                annots_df = pd.read_csv(annots_path)
                results_df = pd.read_csv(results_path)

                ### rename column
                annots_df.rename(columns={'mask_file': 'filename'}, inplace=True)
                ### remove extension
                results_df['filename'] = annots_df['filename'].str.replace('.png', '')
                
                # Merge based on mask_file
                merged_df = pd.merge(
                    annots_df,
                    results_df[['filename', 'type']],  # Only take filename and type columns
                    left_on='filename',
                    right_on='filename',
                    how='left'
                )
                
                # Clean up merged dataframe
                if 'file' in merged_df.columns:
                    merged_df = merged_df.drop('file', axis=1)
                
                # Save to transformed folder
                output_path = self.second_step_processor.get_transformed_folder_path(folder) / "merged_annotations.csv"
                merged_df.to_csv(output_path, index=False)
                
                return f"Successfully merged annotations with classifications"
                
            except Exception as e:
                print(f"Error merging annotations: {str(e)}")
                return f"Error merging annotations: {str(e)}"
            
        def handle_folder_change(folder):
            """Handle folder selection without running the model"""
            if not folder:
                return {
                    status_text: "Please select a folder",
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }
            
            try:
                # Try to load existing results first
                results = self.second_step_processor.load_results(folder)
                
                if not results.empty:
                    # If we have results, load the first image
                    first_row = results.iloc[0]
                    original_path = self.second_step_processor.get_original_path(folder, first_row['filename'])
                    transformed_path = self.second_step_processor.get_transformed_path(folder, first_row['filename'])
                    
                    return {
                        status_text: f"Loaded folder with {len(results)} processed images",
                        original_image: str(original_path),
                        transformed_image: str(transformed_path),
                        type_dropdown: first_row['type'],
                        image_counter: 0
                    }
                else:
                    # If no results yet, just load the first original image
                    source_folder = self.pred_output_dir / folder
                    image_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
                    
                    if image_files:
                        original_path = self.second_step_processor.get_original_path(folder, image_files[0])
                        return {
                            status_text: f"Found {len(image_files)} images to process",
                            original_image: str(original_path),
                            transformed_image: None,
                            type_dropdown: None,
                            image_counter: 0
                        }
                    else:
                        return {
                            status_text: "No images found in folder",
                            original_image: None,
                            transformed_image: None,
                            type_dropdown: None,
                            image_counter: 0
                        }
                    
            except Exception as e:
                error_msg = f"Error loading folder: {str(e)}"
                print(error_msg)
                return {
                    status_text: error_msg,
                    original_image: None,
                    transformed_image: None,
                    type_dropdown: None,
                    image_counter: 0
                }
        def validate_acronym(acronym: str) -> str:
                """Validate the acronym format"""
                if not acronym:
                    return "Please enter an acronym"
                if not acronym.replace('_', '').isalnum():
                    return "Acronym can only contain letters, numbers, and underscores"
                return ""
            
        def handle_export_click(folder: str):
                """Show export dialog when export button is clicked"""
                if not folder:
                    return {
                        export_dialog: gr.update(visible=False),
                        status_text: "Please select a folder first"
                    }
                return {
                    export_dialog: gr.update(visible=True),
                    acronym_input: "",
                    export_status: ""
                }
            
        def handle_export_confirm(folder: str, acronym: str):
                """Handle the export confirmation"""
                # Validate acronym
                validation_msg = validate_acronym(acronym)
                if validation_msg:
                    return {
                        export_status: validation_msg,
                        export_dialog: gr.update(visible=True),
                        status_text: validation_msg
                    }
                
                # Process export
                result = self.export_processor.export_results(folder, acronym)
                
                return {
                    export_status: result,
                    export_dialog: gr.update(visible=False),
                    status_text: result
                }
        
        # Connect event handlers
        process_button.click(
            fn=process_with_options,
            inputs=[folder_dropdown, auto_flip_vertical, auto_flip_horizontal],
            outputs=[status_text, original_image, transformed_image, 
                    type_dropdown, image_counter]
        )
        
        flip_vertical_btn.click(
            fn=lambda f, i: manual_flip(f, i, "vertical"),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, status_text]
        )
        
        flip_horizontal_btn.click(
            fn=lambda f, i: manual_flip(f, i, "horizontal"),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, status_text]
        )
        
        next_button.click(
            fn=lambda f, i: navigate_images(f, "next", i),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, type_dropdown,
                    image_counter, status_text]
        )
        
        prev_button.click(
            fn=lambda f, i: navigate_images(f, "prev", i),
            inputs=[folder_dropdown, image_counter],
            outputs=[original_image, transformed_image, type_dropdown,
                    image_counter, status_text]
        )
        
        type_dropdown.change(
            fn=update_type,
            inputs=[folder_dropdown, image_counter, type_dropdown],
            outputs=status_text
        )
        
        folder_dropdown.change(
            fn=handle_folder_change,
            inputs=[folder_dropdown],
            outputs=[status_text, original_image, transformed_image, 
                    type_dropdown, image_counter]
        )
       

        merge_button.click(
                fn=merge_annotations,
                inputs=[folder_dropdown],
                outputs=[status_text]
            )
        
        return {
            "folder_dropdown": folder_dropdown
        }
    


    def _create_export_dialog(self) -> tuple:
            """Create the export dialog with improved styling"""
            with gr.Group(visible=False) as dialog:
                with gr.Group():
                    with gr.Column():
                        # Header Section
                        gr.HTML(
                            """
                            <div style="text-align: center; padding: 1em; background-color: #f8f9fa; 
                                    border-radius: 8px; margin-bottom: 1em;">
                                <h2 style="margin: 0; color: #2c3e50;">📦 Export Options</h2>
                                <p style="margin: 0.5em 0 0 0; color: #7f8c8d;">
                                    Configure your export settings
                                </p>
                            </div>
                            """
                        )
                        
                        # Basic Export Settings
                        with gr.Group():
                            gr.HTML(
                                """
                                <div style="padding: 1em; border-radius: 8px;">
                                    <h3 style="margin-bottom: 1em; color: #7f8c8d;">
                                        🏷️ Basic Settings
                                    </h3>
                                </div>
                                """
                            )
                            acronym_input = gr.Textbox(
                                label="Export Acronym",
                                placeholder="Enter acronym (e.g., OSA_2024)...",
                                info="Only letters, numbers, and underscores allowed",
                                scale=1
                            )
                        
                        # PDF Export Options
                        with gr.Group():
                            gr.HTML(
                                """
                                <div style="padding: 1em; border-radius: 8px;">
                                    <h3 style="margin-bottom: 1em; color: #7f8c8d;">
                                        📄 PDF Catalog Options
                                    </h3>
                                </div>
                                """
                            )
                            with gr.Row():
                                pdf_export = gr.Checkbox(
                                    label="Generate PDF Catalog",
                                    value=False,
                                    info="Create a PDF catalog with all exported images",
                                    scale=1
                                )
                            
                            with gr.Column(visible=False) as pdf_options:
                                with gr.Row():
                                    page_size = gr.Dropdown(
                                        label="Page Size",
                                        choices=['A4', 'A3', 'A5', 'LETTER', 'LEGAL'],
                                        value='A4',
                                        info="Select the PDF page size",
                                        scale=1
                                    )
                                with gr.Row():
                                    scale_factor = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05,
                                        label="Image Scale Factor",
                                        info="Adjust the size of images in the PDF",
                                        scale=1
                                    )
                        
                        # Status and Buttons
                        with gr.Group():
                            export_status = gr.Text(
                                label="Status",
                                interactive=False,
                                show_label=False
                            )
                            
                            with gr.Row():
                                cancel_btn = gr.Button(
                                    "❌ Cancel", 
                                    variant="secondary",
                                    scale=1
                                )
                                export_btn = gr.Button(
                                    "📦 Export", 
                                    variant="primary",
                                    scale=1
                                )
                        
                        # Show/hide PDF options based on checkbox
                        pdf_export.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[pdf_export],
                            outputs=[pdf_options]
                        )
                        
            return dialog, acronym_input, export_status, cancel_btn, export_btn, pdf_export, page_size, scale_factor
import torch
import psutil
import platform
import os
from datetime import datetime
import sys
from pathlib import Path
import GPUtil

def get_size(bytes):
    """
    Convert bytes to human readable format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

def get_system_info():
    """Get system information including CPU, RAM, and GPU"""
    # [Previous get_system_info implementation remains the same]
    pass

def print_ascii_banner():
    banner = """
 ____         ____       _   _                  _                    
|  _ \ _   _ |  _ \ ___ | |_| |_ ___ _ __ _   | |    ___ _ __  ___ 
| |_) | | | || |_) / _ \| __| __/ _ \ '__| | | | |   / _ \ '_ \/ __|
|  __/| |_| ||  __/ (_) | |_| ||  __/ |  | |_| | |__|  __/ | | \__ \\
|_|    \__, ||_|   \___/ \__|\__\___|_|   \__, |_____\___|_| |_|___/
       |___/                               |___/                      
                                                                  
    🏺 V0.1 🔍
"""
    return banner


def print_startup_banner():
    """
    Print a beautiful startup banner with system information
    """
    # Get terminal width for centered text
    term_width = 80
    
    
    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get system information
    sys_info = get_system_info()
    
    # Print banner
    print("\n" + "="*term_width)
    print(print_ascii_banner())
    print("="*term_width)
    print(f"\n🕒 Start Time: {current_time}")
    
    if sys_info:
        print(f"\n💻 System Information:")
        print(f"   OS: {sys_info['os']}")
        print(f"   Python: {sys_info['python']}")
        print(f"   CPU: {sys_info['cpu']}")
        print(f"\n💾 Memory Status:")
        print(f"   Total: {sys_info['ram']['total']}")
        print(f"   Used: {sys_info['ram']['used']} ({sys_info['ram']['percent']}%)")
        print(f"   Available: {sys_info['ram']['available']}")
        
        if sys_info['gpu']:
            print(f"\n🎮 GPU Information:")
            for i, gpu in enumerate(sys_info['gpu']):
                if 'memory_total' in gpu:
                    print(f"   GPU {i+1}: {gpu['name']}")
                    print(f"   Memory: {gpu['memory_used']}/{gpu['memory_total']}")
                    print(f"   Load: {gpu['load']}")
                else:
                    print(f"   GPU {i+1}: {gpu['name']}")
    
    print("\n📂 Directory Structure:")
    required_dirs = ["outputs", "pdf2img_outputs", "models_vision", "models_classifier"]
    for dir_name in required_dirs:
        status = "✅" if os.path.exists(dir_name) else "❌"
        print(f"   {status} {dir_name}")
    
    print("\n🚀 Initialization:")
    print("   ✅ Loading components...")


def print_version_info():
    """Print comprehensive version information for all dependencies"""
    
    # Define packages to check, grouped by category
    packages = {
        "Core Dependencies": [
            ("PyTorch", "torch"),
            ("Gradio", "gradio"),
            ("NumPy", "numpy"),
            ("Pandas", "pandas")
        ],
        "Computer Vision": [
            ("Ultralytics", "ultralytics"),
            ("PIL/Pillow", "PIL"),
            ("scikit-image", "skimage"),
            ("OpenCV", "cv2")
        ],
        "PDF Processing": [
            ("PyMuPDF", "fitz"),
            ("ReportLab", "reportlab")
        ],
        "Deep Learning": [
            ("timm", "timm"),
            ("torchvision", "torchvision")
        ],
        "Scientific Computing": [
            ("SciPy", "scipy"),
            #("scikit-learn", "sklearn")
        ]
    }

    def get_package_version(package_name):
        """Get package version with error handling"""
        try:
            module = __import__(package_name)
            try:
                return module.__version__
            except AttributeError:
                if hasattr(module, 'PIL_VERSION'):  # Special case for PIL
                    return module.PIL_VERSION
                elif hasattr(module, 'VERSION'):  # Some packages use VERSION
                    return module.VERSION
                elif hasattr(module, 'version'):  # Some packages use version
                    return module.version
                else:
                    return "Version unknown"
        except ImportError:
            return "Not installed"

    # Print header
    print("\n" + "="*50)
    print("📦 Dependencies Information")
    print("="*50)

    # Print system information first
    import platform
    import sys
    print(f"\n🖥️  System Information:")
    print(f"   • Python: {sys.version.split()[0]}")
    print(f"   • Platform: {platform.platform()}")
    if platform.system() == "Windows":
        print(f"   • Windows Version: {platform.win32_ver()[0]}")

    # Print CUDA information if available
    if torch.cuda.is_available():
        print(f"\n🎮 CUDA Information:")
        print(f"   • CUDA Available: Yes")
        print(f"   • CUDA Version: {torch.version.cuda}")
        print(f"   • cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   • Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   • GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"\n🎮 CUDA Information:")
        print(f"   • CUDA Available: No")

    # Print package versions by category
    for category, package_list in packages.items():
        print(f"\n{category}:")
        for package_display_name, package_import_name in package_list:
            version = get_package_version(package_import_name)
            status_icon = "✅" if version != "Not installed" else "❌"
            print(f"   {status_icon} {package_display_name}: {version}")

    print("\n" + "="*50)
    
    # Print warnings for critical missing packages
    critical_packages = ["torch", "gradio", "fitz", "ultralytics"]
    missing_critical = [pkg for pkg in critical_packages 
                       if get_package_version(pkg) == "Not installed"]
    
    if missing_critical:
        print("\n⚠️ Warning: Critical packages missing:")
        for pkg in missing_critical:
            print(f"   • {pkg}")
        print("Please install these packages for full functionality.")
        
    print("\nℹ️ To install missing packages, use:")
    print("   pip install package_name")
    print("="*50 + "\n")


if __name__ == "__main__":
    try:
        print_startup_banner()
        print_version_info()
        
        # Initialize app
        print("\n   ✅ Initializing PyPotteryLens...")
        app = App()
        demo = app.build_interface()
        
        print("\n✨ PyPotteryLens is ready!")
        print("🌐 Opening browser window...")
        print("📝 If the browser doesn't open automatically, visit: http://localhost:7860")
        print("\n" + "="*80 + "\n")
        
        # Launch the application
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
        )
        
    except Exception as e:
        print("\n❌ Error during startup:")
        print(f"   {str(e)}")
        sys.exit(1)