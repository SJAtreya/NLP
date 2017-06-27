import tornado.ioloop
import tornado.web
import classifier as Classifier

class MainHandler(tornado.web.RequestHandler):
    def post(self):
        retVal = Classifier.classify(self.get_argument("query",None, True))
        print (retVal)
        self.write(retVal)

class OrderHandler(tornado.web.RequestHandler):
    def post(self):
        if self.get_argument("query","No", True) == 'yes':
            self.write({'message':"Order has been sent to the pharmacy. I will keep you posted when the order is processed.", 'status':"Success", 'callback':"/"})
        else:
            self.write({'message':"Okay. Please send the order before it's cut-off time.", 'status':"Failed", 'callback':"/"})
		
def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
		(r"/order", OrderHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
    print("Server Listening....")
