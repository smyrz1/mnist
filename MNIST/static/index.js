(function()
{
	var canvas = document.querySelector( "#canvas" );
	var context = canvas.getContext( "2d" );
	canvas.width = 280;
	canvas.height = 280;

	var flag=false
	context.fillStyle="white";
	context.fillRect(0,0,canvas.width,canvas.height);
	context.color = "black";
	context.lineWidth = 10;
    context.lineJoin = context.lineCap = 'round';

	canvas.onmousedown=function(evt){

				var BeginX=evt.clientX-this.offsetLeft;
				var BeginY=evt.clientY-this.offsetTop;
				context.beginPath();
				context.moveTo(BeginX, BeginY);
				flag=true;

		}

	canvas.onmousemove=function(evt){
			if(flag){
				var endX=evt.clientX-this.offsetLeft;
				var endY=evt.clientY-this.offsetTop;
				context.lineTo(endX, endY);
				context.stroke();
			}
		}

	canvas.onmouseup=function(){
			flag=false;
		}
	canvas.onmouseout=function(){
			flag=false;
		}

}());