import AssemblyKeys._

name := "sann"

description := "Simple Artificial Neural Network"

organization := "org.sann"

version := "0.1-SNAPSHOT"

scalaVersion := "2.11.4"

libraryDependencies ++= {
  val breezeVersion = "0.10"
  Seq(
    "org.scalanlp" %% "breeze" % breezeVersion,
    "org.scalanlp" %% "breeze-natives" % breezeVersion,
    "org.scalanlp" %% "breeze-viz" % "0.9",
    "org.testng" % "testng" % "6.1.1" % "test"
  )
}

assemblySettings

jarName in assembly := "sann.jar"