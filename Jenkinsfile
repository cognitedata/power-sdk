@Library('jenkins-helpers') _
testBuildReleasePoetryPackage {
    releaseToArtifactory = true
    testWithTox = false
    toxEnvList = ['py36', 'py37', 'py38']
    beforeTests = {
        stage('Build Docs'){
            dir('./docs'){
                sh("poetry run sphinx-build -W -b html ./source ./build")
            }
        }
    }
}