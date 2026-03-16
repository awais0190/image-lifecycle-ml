module.exports = {
  apps: [{
    name: 'ml-service',
    script: '/home/omar-akhter/MY-Projects/awais_chatha_project/image-lifecycle-ml/venv/bin/uvicorn',
    args: 'app:app --host 0.0.0.0 --port 8000',
    cwd: '/home/omar-akhter/MY-Projects/awais_chatha_project/image-lifecycle-ml',
    interpreter: 'none',
    restart_delay: 3000,
    max_restarts: 20,
    autorestart: true,
    env: {
      PATH: '/home/omar-akhter/MY-Projects/awais_chatha_project/image-lifecycle-ml/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
      VIRTUAL_ENV: '/home/omar-akhter/MY-Projects/awais_chatha_project/image-lifecycle-ml/venv',
    },
  }],
};
