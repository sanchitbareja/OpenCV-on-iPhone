#import "OpenCVTestViewController.h"

#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/objdetect/objdetect.hpp>

#import "SHK.h"

@implementation OpenCVTestViewController
@synthesize imageView, process, instructions, processingtype;

- (void)dealloc {
	AudioServicesDisposeSystemSoundID(alertSoundID);
	[imageView dealloc];
	[super dealloc];
}

#pragma mark -
#pragma mark OpenCV Support Methods

// NOTE you SHOULD cvReleaseImage() for the return value when end of the code.
- (IplImage *)CreateIplImageFromUIImage:(UIImage *)image {
	CGImageRef imageRef = image.CGImage;

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	IplImage *iplimage = cvCreateImage(cvSize(image.size.width, image.size.height), IPL_DEPTH_8U, 4);
	CGContextRef contextRef = CGBitmapContextCreate(iplimage->imageData, iplimage->width, iplimage->height,
													iplimage->depth, iplimage->widthStep,
													colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
	CGContextDrawImage(contextRef, CGRectMake(0, 0, image.size.width, image.size.height), imageRef);
	CGContextRelease(contextRef);
	CGColorSpaceRelease(colorSpace);

	IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
	cvCvtColor(iplimage, ret, CV_RGBA2BGR);
	cvReleaseImage(&iplimage);

	return ret;
}

// NOTE You should convert color mode as RGB before passing to this function
- (UIImage *)UIImageFromIplImage:(IplImage *)image {
	NSLog(@"IplImage (%d, %d) %d bits by %d channels, %d bytes/row %s", image->width, image->height, image->depth, image->nChannels, image->widthStep, image->channelSeq);

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	NSData *data = [NSData dataWithBytes:image->imageData length:image->imageSize];
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
	CGImageRef imageRef = CGImageCreate(image->width, image->height,
										image->depth, image->depth * image->nChannels, image->widthStep,
										colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
										provider, NULL, false, kCGRenderingIntentDefault);
	UIImage *ret = [UIImage imageWithCGImage:imageRef];
	CGImageRelease(imageRef);
	CGDataProviderRelease(provider);
	CGColorSpaceRelease(colorSpace);
	return ret;
}

#pragma mark -
#pragma mark Utilities for intarnal use

- (void)showProgressIndicator:(NSString *)text {
	//[UIApplication sharedApplication].networkActivityIndicatorVisible = YES;
	self.view.userInteractionEnabled = FALSE;
	if(!progressHUD) {
		CGFloat w = 160.0f, h = 120.0f;
		progressHUD = [[UIProgressHUD alloc] initWithFrame:CGRectMake((self.view.frame.size.width-w)/2, (self.view.frame.size.height-h)/2, w, h)];
		[progressHUD setText:text];
		[progressHUD showInView:self.view];
	}
}

- (void)hideProgressIndicator {
	//[UIApplication sharedApplication].networkActivityIndicatorVisible = NO;
	self.view.userInteractionEnabled = TRUE;
	if(progressHUD) {
		[progressHUD hide];
		[progressHUD release];
		progressHUD = nil;

		AudioServicesPlaySystemSound(alertSoundID);
	}
}


- (void) opencvFaceDetect:(UIImage *)overlayImage  {
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	if(imageView.image) {
		cvSetErrMode(CV_ErrModeParent);

		IplImage *image = [self CreateIplImageFromUIImage:imageView.image];
		
		// Scaling down
		IplImage *small_image = cvCreateImage(cvSize(image->width/2,image->height/2), IPL_DEPTH_8U, 3);
		cvPyrDown(image, small_image, CV_GAUSSIAN_5x5);
		int scale = 2;
		
		// Load XML
		NSString *path = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_default" ofType:@"xml"];
		CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*)cvLoad([path cStringUsingEncoding:NSASCIIStringEncoding], NULL, NULL, NULL);
		CvMemStorage* storage = cvCreateMemStorage(0);
		
		// Detect faces and draw rectangle on them
		CvSeq* faces = cvHaarDetectObjects(small_image, cascade, storage, 1.2f, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(0,0), cvSize(20, 20));
		cvReleaseImage(&small_image);
		
		// Create canvas to show the results
		CGImageRef imageRef = imageView.image.CGImage;
		CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
		CGContextRef contextRef = CGBitmapContextCreate(NULL, imageView.image.size.width, imageView.image.size.height,
														8, imageView.image.size.width * 4,
														colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
		CGContextDrawImage(contextRef, CGRectMake(0, 0, imageView.image.size.width, imageView.image.size.height), imageRef);
		
		CGContextSetLineWidth(contextRef, 4);
		CGContextSetRGBStrokeColor(contextRef, 0.0, 0.0, 1.0, 0.5);
		
		// Draw results on the iamge
		for(int i = 0; i < faces->total; i++) {
			NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
			
			// Calc the rect of faces
			CvRect cvrect = *(CvRect*)cvGetSeqElem(faces, i);
			CGRect face_rect = CGContextConvertRectToDeviceSpace(contextRef, CGRectMake(cvrect.x * scale, cvrect.y * scale, cvrect.width * scale, cvrect.height * scale));
			
			if(overlayImage) {
				CGContextDrawImage(contextRef, face_rect, overlayImage.CGImage);
			} else {
				CGContextStrokeRect(contextRef, face_rect);
			}
			
			[pool release];
		}
		
		imageView.image = [UIImage imageWithCGImage:CGBitmapContextCreateImage(contextRef)];
		CGContextRelease(contextRef);
		CGColorSpaceRelease(colorSpace);
		
		cvReleaseMemStorage(&storage);
		cvReleaseHaarClassifierCascade(&cascade);

		[self hideProgressIndicator];
	}

	[pool release];
}


- (void)opencvEdgeDetectManual {
    
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    
	if(imageView.image) {
		cvSetErrMode(CV_ErrModeParent);
        /*
        if([imageView.subviews count] < 4){
            NSArray *tl = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:imageView.image.size.width/4],[NSNumber numberWithFloat:imageView.image.size.height/4], nil];
            NSArray *tr = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:imageView.image.size.width*3/4],[NSNumber numberWithFloat:imageView.image.size.height/4], nil];
            NSArray *bl = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:imageView.image.size.width/4],[NSNumber numberWithFloat:imageView.image.size.height*3/4], nil];
            NSArray *br = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:imageView.image.size.width*3/4],[NSNumber numberWithFloat:imageView.image.size.height*3/4], nil];
            
            //[points removeAllObjects];
            [points addObject:tl];
            [points addObject:tr];
            [points addObject:bl];
            [points addObject:br];
            
        } else {
            UIView* view1 = [[NSArray arrayWithArray:imageView.subviews] objectAtIndex:0];
            UIView* view2 = [[NSArray arrayWithArray:imageView.subviews] objectAtIndex:1];
            UIView* view3 = [[NSArray arrayWithArray:imageView.subviews] objectAtIndex:2];
            UIView* view4 = [[NSArray arrayWithArray:imageView.subviews] objectAtIndex:3];
            NSLog(@"%@,%@,%@,%@",view1,view2,view3,view4);
            
            NSArray *point1 = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:view1.center.x],[NSNumber numberWithFloat:view1.center.y], nil];
            NSArray *point2 = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:view2.center.x],[NSNumber numberWithFloat:view2.center.y], nil];
            NSArray *point3 = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:view3.center.x],[NSNumber numberWithFloat:view3.center.y], nil];
            NSArray *point4 = [[NSArray alloc] initWithObjects:[NSNumber numberWithFloat:view4.center.x],[NSNumber numberWithFloat:view4.center.y], nil];
            
            //[points removeAllObjects];
            [points addObject:point1];
            [points addObject:point2];
            [points addObject:point3];
            [points addObject:point4];
            
            [view1 release];
            [view2 release];
            [view3 release];
            [view4 release];
        }
        */
        
		// Create grayscale IplImage from UIImage
		IplImage *img_color = [self CreateIplImageFromUIImage:imageView.image];
		IplImage *img = cvCreateImage(cvGetSize(img_color), IPL_DEPTH_8U, 1);
		cvCvtColor(img_color, img, CV_BGR2GRAY);
		
		// Detect edge
        //OpenCV documentation of functions: http://opencv.willowgarage.com/documentation/feature_detection.html
        
        //equalize histogram
        IplImage *img_temp = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvEqualizeHist(img,img_temp);
        
		//Canny
        IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
		cvCanny(img_temp, img2, 300, 300, 3);
        
        /*
        //get subimage
        //top left quad
        float pt1x = [[[points objectAtIndex:0] objectAtIndex:0] floatValue];
        float pt1y = [[[points objectAtIndex:0] objectAtIndex:1] floatValue];
        cvSetImageROI(img2, cvRect(pt1x, 
                                   pt1y, 
                                   20, 
                                   20));
        IplImage *topleft = cvCreateImage(cvSize(20, 20), IPL_DEPTH_8U, 1);
        cvCopy(img2, topleft, NULL);
        cvResetImageROI(img2);
        //top right quad
        float pt2x = [[[points objectAtIndex:1] objectAtIndex:0] floatValue];
        float pt2y = [[[points objectAtIndex:1] objectAtIndex:1] floatValue];
        cvSetImageROI(img2, cvRect(pt2x,
                                   pt2y,
                                   20, 
                                   20));
        IplImage *topright = cvCreateImage(cvSize(20, 20), IPL_DEPTH_8U, 1);
        cvCopy(img2, topright, NULL);
        cvResetImageROI(img2);
        //bottom left quad
        float pt3x = [[[points objectAtIndex:2] objectAtIndex:0] floatValue];
        float pt3y = [[[points objectAtIndex:2] objectAtIndex:1] floatValue];
        cvSetImageROI(img2, cvRect(pt3x,
                                   pt3y, 
                                   20, 
                                   20));
        IplImage *bottomleft = cvCreateImage(cvSize(20, 20), IPL_DEPTH_8U, 1);
        cvCopy(img2, bottomleft, NULL);
        cvResetImageROI(img2);
        //bottom right quad
        float pt4x = [[[points objectAtIndex:3] objectAtIndex:0] floatValue];
        float pt4y = [[[points objectAtIndex:3] objectAtIndex:1] floatValue];
        cvSetImageROI(img2, cvRect(pt4x,
                                   pt4y, 
                                   20, 
                                   20));
        IplImage *bottomright = cvCreateImage(cvSize(20, 20), IPL_DEPTH_8U, 1);
        cvCopy(img2, bottomright, NULL);
        cvResetImageROI(img2);
        
        //IplImage *img3 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_32F, 1);        
        //cvCornerHarris(img2, img3, 1, 5, 0.01);
        //cvPreCornerDetect(img2, img3, 5);
        
        //cvConvertScale(img3, img2,256,0); 
        */
        
        //Hough
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* lines = 0;
        
        lines = cvHoughLines2( img2,
                              storage,
                              CV_HOUGH_STANDARD,
                              1,
                              CV_PI/180,
                              120,
                              0,
                              0);
        
        for(int i = 0; i < lines->total; i++ )
        {
            float* line = (float*)cvGetSeqElem(lines,i);
            float rho = line[0];
            float theta = line[1];
            CvPoint pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            cvLine( img2, pt1, pt2, CV_RGB(255,255,255), 1, CV_AA,0 );
            //NSLog(@"rho:%f , theta:%f ",rho, theta);
            //NSLog(@"pt1.x:%d.,pt1.y:%d / pt2.x:%d,pt2.y:%d",pt1.x,pt1.y,pt2.x,pt2.y);
        }
        
        NSLog(@"imgwidth:%d,imgheight:%d",img2->width,img2->height);
        // NSLog(@"imgwidth:%d,imgheight:%d",img3->width,img3->height);
        NSLog(@"lines:%d",lines->total);
        
        
        
        //CvPerspectiveTranform Not working properly
        CvMat* mmat = cvCreateMat(3,3,CV_32FC1);
        //CvPoint2D32f p1 = cvPoint2D32f(43.0, 18.0);
        
        CvPoint2D32f c1 = (cvPoint2D32f(200, 18),cvPoint2D32f(320,100),cvPoint2D32f(10, 253),cvPoint2D32f(224, 200));
        CvPoint2D32f c2 = (cvPoint2D32f(0, 0),cvPoint2D32f(320,0),cvPoint2D32f(0, 240),cvPoint2D32f(320, 240));
        
        //change image size where applicable
        IplImage *img3 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);
        
        //mmat = cvGetPerspectiveTransform(&c1, &c2, mmat);
        cvmSet(mmat, 0, 0, 1.0);
        cvmSet(mmat, 0, 1, 1.0);
        cvmSet(mmat, 0, 2, 1.0);
        cvmSet(mmat, 1, 0, 1.0);
        cvmSet(mmat, 1, 1, 1.0);
        cvmSet(mmat, 1, 2, 1.0);
        cvmSet(mmat, 2, 0, 0.2);
        cvmSet(mmat, 2, 1, 0.2);
        cvmSet(mmat, 2, 2, 0.2);
        
        NSLog(@"0,0:%f",cvmGet(mmat, 0, 0));
        NSLog(@"0,1:%f",cvmGet(mmat, 0, 1));
        NSLog(@"0,2:%f",cvmGet(mmat, 0, 2));
        NSLog(@"1,0:%f",cvmGet(mmat, 1, 0));
        NSLog(@"1,1:%f",cvmGet(mmat, 1, 1));
        NSLog(@"1,2:%f",cvmGet(mmat, 1, 2));
        NSLog(@"2,0:%f",cvmGet(mmat, 2, 0));
        NSLog(@"2,1:%f",cvmGet(mmat, 2, 1));
        NSLog(@"2,2:%f",cvmGet(mmat, 2, 2));
        
        //cvWarpPerspective(img2, img3, mmat,CV_WARP_INVERSE_MAP,cvScalarAll(0));
        //cvLogPolar( img, img3, cvPoint2D32f(img2->width/2,img2->height/2), 10,
          //         CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP );
        //cvGetPerspectiveTransform(c1, c2, mmat);
        
        // Convert black and white to 24bit image then convert to UIImage to show
		IplImage *returnimage = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
		for(int y=0; y<img2->height; y++) {
			for(int x=0; x<img2->width; x++) {
				char *p = returnimage->imageData + y * returnimage->widthStep + x*3;
				*p = *(p+1) = *(p+2) = img2->imageData[y * img2->widthStep + x];
			}
		}
		
        UIImage *image = [self UIImageFromIplImage:returnimage];
        
        NSLog(@"returnImage width:%f, height:%f",image.size.width,image.size.height);
        
		imageView.image = image;
        
        cvReleaseImage(&img);
        cvReleaseImage(&img2);
        cvReleaseImage(&img3);
        //cvReleaseImage(&topleft);
        //cvReleaseImage(&topright);
        //cvReleaseImage(&bottomleft);
        //cvReleaseImage(&bottomright);
        cvReleaseImage(&img_color);
		cvReleaseImage(&returnimage);

        /*
        [points removeAllObjects];            
        for (UIView *view in imageView.subviews) {
            [view removeFromSuperview];
        }
        */
		
        [self hideProgressIndicator];
	}
    
    instructions.text = @"Done!";
    
	[pool release];
}

#pragma mark -
#pragma mark IBAction

- (IBAction)loadImage:(id)sender {
	if(!actionSheetAction) {
		UIActionSheet *actionSheet = [[UIActionSheet alloc] initWithTitle:@""
																 delegate:self cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil
														otherButtonTitles:@"Use Photo from Library", @"Take Photo with Camera", @"Use Default Lena", nil];
		actionSheet.actionSheetStyle = UIActionSheetStyleDefault;
		actionSheetAction = ActionSheetToSelectTypeOfSource;
		[actionSheet showInView:self.view];
		[actionSheet release];
	}
}

- (IBAction)saveImage:(id)sender {
	if(imageView.image) {
		[self showProgressIndicator:@"Saving"];
		UIImageWriteToSavedPhotosAlbum(imageView.image, self, @selector(finishUIImageWriteToSavedPhotosAlbum:didFinishSavingWithError:contextInfo:), nil);
	}
}

- (void)finishUIImageWriteToSavedPhotosAlbum:(UIImage *)image didFinishSavingWithError:(NSError *)error contextInfo:(void *)contextInfo {
	[self hideProgressIndicator];
}

- (IBAction)edgeDetect:(id)sender {
	[self showProgressIndicator:@"Detecting"];
    instructions.text = @"Processing...";
    [self performSelectorInBackground:@selector(opencvEdgeDetectManual) withObject:nil];
}

- (IBAction)faceDetect:(id)sender {
	cvSetErrMode(CV_ErrModeParent);
	if(imageView.image && !actionSheetAction) {
		UIActionSheet *actionSheet = [[UIActionSheet alloc] initWithTitle:@""
																 delegate:self cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil
														otherButtonTitles:@"Bounding Box", @"Laughing Man", nil];
		actionSheet.actionSheetStyle = UIActionSheetStyleDefault;
		actionSheetAction = ActionSheetToSelectTypeOfMarks;
		[actionSheet showInView:self.view];
		[actionSheet release];
	}
}

- (IBAction)share:(id)sender {
    //Sharing images on facebook/twitter.
    //Create the item to share (in this example, a url)
    UIImage *image = imageView.image;
    SHKItem *item = [SHKItem image:image title:@"Share Homework on iPhone!"];
    
    // Get the ShareKit action sheet
    SHKActionSheet *actionSheet = [SHKActionSheet actionSheetForItem:item];
     
    // Display the action sheet
    [actionSheet showInView:self.view];
     
    [SHK flushOfflineQueue];
}

-(IBAction) segmentedControlIndexChanged{
    switch (processingtype.selectedSegmentIndex) {
        case 0:
            instructions.text =@"Segment 1 selected.";
            process.enabled = YES;
            break;
        case 1:
            instructions.text =@"Segment 2 selected.";
            if([imageView.subviews count] >= 4){
                process.enabled = YES;
            } else {
                process.enabled = NO;
            }
            break;
        default:
            break;
    }
}


#pragma mark -
#pragma mark UIViewControllerDelegate

- (void)viewDidLoad {
	[super viewDidLoad];
	[[UIApplication sharedApplication] setStatusBarStyle:UIStatusBarStyleBlackOpaque animated:YES];
	points = [[NSMutableArray alloc] init];
    [self loadImage:nil];

	NSURL *url = [NSURL fileURLWithPath:[[NSBundle mainBundle] pathForResource:@"Tink" ofType:@"aiff"] isDirectory:NO];
	AudioServicesCreateSystemSoundID((CFURLRef)url, &alertSoundID);
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
	return NO;
}

#pragma mark -
#pragma mark UIActionSheetDelegate

- (void)actionSheet:(UIActionSheet *)actionSheet clickedButtonAtIndex:(NSInteger)buttonIndex {
	switch(actionSheetAction) {
		case ActionSheetToSelectTypeOfSource: {
			UIImagePickerControllerSourceType sourceType;
			if (buttonIndex == 0) {
				sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
			} else if(buttonIndex == 1) {
				sourceType = UIImagePickerControllerSourceTypeCamera;
			} else if(buttonIndex == 2) {
				NSString *path = [[NSBundle mainBundle] pathForResource:@"lena" ofType:@"jpg"];
				imageView.image = [UIImage imageWithContentsOfFile:path];
				break;
			} else {
				// Cancel
				break;
			}
			if([UIImagePickerController isSourceTypeAvailable:sourceType]) {
				UIImagePickerController *picker = [[UIImagePickerController alloc] init];
				picker.sourceType = sourceType;
				picker.delegate = self;
				picker.allowsImageEditing = NO;
				[self presentModalViewController:picker animated:YES];
				[picker release];
			}
			break;
		}
		case ActionSheetToSelectTypeOfMarks: {
			if(buttonIndex != 0 && buttonIndex != 1) {
				break;
			}

			UIImage *image = nil;
			if(buttonIndex == 1) {
				NSString *path = [[NSBundle mainBundle] pathForResource:@"laughing_man" ofType:@"png"];
				image = [UIImage imageWithContentsOfFile:path];
			}

			[self showProgressIndicator:@"Detecting"];
			[self performSelectorInBackground:@selector(opencvFaceDetect:) withObject:image];
			break;
		}
	}
	actionSheetAction = 0;
}

#pragma mark -
#pragma mark UIImagePickerControllerDelegate

- (UIImage *)scaleAndRotateImage:(UIImage *)image {
	static int kMaxResolution = 640;
	
	CGImageRef imgRef = image.CGImage;
	CGFloat width = CGImageGetWidth(imgRef);
	CGFloat height = CGImageGetHeight(imgRef);
	
	CGAffineTransform transform = CGAffineTransformIdentity;
	CGRect bounds = CGRectMake(0, 0, width, height);
	if (width > kMaxResolution || height > kMaxResolution) {
		CGFloat ratio = width/height;
		if (ratio > 1) {
			bounds.size.width = kMaxResolution;
			bounds.size.height = bounds.size.width / ratio;
		} else {
			bounds.size.height = kMaxResolution;
			bounds.size.width = bounds.size.height * ratio;
		}
	}
	
	CGFloat scaleRatio = bounds.size.width / width;
	CGSize imageSize = CGSizeMake(CGImageGetWidth(imgRef), CGImageGetHeight(imgRef));
	CGFloat boundHeight;
	
	UIImageOrientation orient = image.imageOrientation;
	switch(orient) {
		case UIImageOrientationUp:
			transform = CGAffineTransformIdentity;
			break;
		case UIImageOrientationUpMirrored:
			transform = CGAffineTransformMakeTranslation(imageSize.width, 0.0);
			transform = CGAffineTransformScale(transform, -1.0, 1.0);
			break;
		case UIImageOrientationDown:
			transform = CGAffineTransformMakeTranslation(imageSize.width, imageSize.height);
			transform = CGAffineTransformRotate(transform, M_PI);
			break;
		case UIImageOrientationDownMirrored:
			transform = CGAffineTransformMakeTranslation(0.0, imageSize.height);
			transform = CGAffineTransformScale(transform, 1.0, -1.0);
			break;
		case UIImageOrientationLeftMirrored:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeTranslation(imageSize.height, imageSize.width);
			transform = CGAffineTransformScale(transform, -1.0, 1.0);
			transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
			break;
		case UIImageOrientationLeft:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeTranslation(0.0, imageSize.width);
			transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
			break;
		case UIImageOrientationRightMirrored:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeScale(-1.0, 1.0);
			transform = CGAffineTransformRotate(transform, M_PI / 2.0);
			break;
		case UIImageOrientationRight:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeTranslation(imageSize.height, 0.0);
			transform = CGAffineTransformRotate(transform, M_PI / 2.0);
			break;
		default:
			[NSException raise:NSInternalInconsistencyException format:@"Invalid image orientation"];
	}
	
	UIGraphicsBeginImageContext(bounds.size);
	CGContextRef context = UIGraphicsGetCurrentContext();
	if (orient == UIImageOrientationRight || orient == UIImageOrientationLeft) {
		CGContextScaleCTM(context, -scaleRatio, scaleRatio);
		CGContextTranslateCTM(context, -height, 0);
	} else {
		CGContextScaleCTM(context, scaleRatio, -scaleRatio);
		CGContextTranslateCTM(context, 0, -height);
	}
	CGContextConcatCTM(context, transform);
	CGContextDrawImage(UIGraphicsGetCurrentContext(), CGRectMake(0, 0, width, height), imgRef);
	UIImage *imageCopy = UIGraphicsGetImageFromCurrentImageContext();
	UIGraphicsEndImageContext();
	
	return imageCopy;
}

- (void)imagePickerController:(UIImagePickerController *)picker
		didFinishPickingImage:(UIImage *)image
				  editingInfo:(NSDictionary *)editingInfo
{
    //remove all rectangles on screen before displaying the next image on screen
    for (UIView *view in imageView.subviews) {
        [view removeFromSuperview];
    }
	imageView.image = [self scaleAndRotateImage:image];
	[[picker parentViewController] dismissModalViewControllerAnimated:YES];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
	[[picker parentViewController] dismissModalViewControllerAnimated:YES];
}

#pragma mark -
#pragma mark touch events

-(void) touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
    if(processingtype.selectedSegmentIndex == 1 && [imageView.subviews count] < 4){
        UITouch *touch = [[event allTouches] anyObject];
        CGPoint location = [touch locationInView:touch.view];
        
        //CGImageRef highlight = CGImageCreateWithImageInRect(imageView.image.CGImage, CGRectMake(location.x-50, location.y-50, 100, 100));
        
        
        UIView *squareView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 50, 50)];
        squareView.backgroundColor = [UIColor darkGrayColor];
        squareView.alpha = 0.3;
        [imageView addSubview:squareView];
        squareView.center = location;
        [squareView release];
        if ([imageView.subviews count] == 4) {
            process.enabled = YES;
            for (int i = 0; i<[imageView.subviews count]; i++) {
                UIView* view1 = [[NSArray arrayWithArray:imageView.subviews] objectAtIndex:i];
                NSLog(@"View1.location.x:%f, View1.location.y:%f",view1.center.x,view1.center.y);
            }
        }
        
    } else if([imageView.subviews count] >= 4) {
        UIAlertView* alert = [[UIAlertView alloc] initWithTitle:@"Next?" message:@"Do you want to process or reselect your corners?" delegate:self cancelButtonTitle:@"Cancel" otherButtonTitles:@"Process",@"Reselect Corners", nil];
        [alert show];
    }
}
/*
-(void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
	[self touchesBegan:touches withEvent:event];
    NSLog(@"Moved");
}
*/
#pragma mark -
#pragma mark UIAlertView delegate

- (void)alertView:(UIAlertView *)alertView clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (buttonIndex == 0) {
        NSLog(@"Cancel button");
    }
    if (buttonIndex == 1) {
        NSLog(@"Process");
        [self performSelectorInBackground:@selector(opencvEdgeDetectManual) withObject:nil];
    }
    if (buttonIndex == 2) {
        NSLog(@"Reselect Corners");
    }
}

@end