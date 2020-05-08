from aiohttp import web

from chatbot.dialog import Classifier


class DialogView(web.View):
    async def post(self):
        params = await self.request.json()
        cl = Classifier()
        data = cl.response(params['sentence'])
        return web.json_response({'response': data})


APP = web.Application()
APP.router.add_view('/api/dialogs/dialog', DialogView)


def main():
    web.run_app(app=APP, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
